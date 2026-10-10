package com.ilche.makeupai

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.text.InputType
import android.view.LayoutInflater
import android.view.View
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.exifinterface.media.ExifInterface
import com.google.android.gms.tasks.Tasks
import com.google.android.material.progressindicator.LinearProgressIndicator
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.face.FaceDetection
import com.google.mlkit.vision.face.FaceDetectorOptions
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import kotlin.math.max

class MainActivity : AppCompatActivity() {

    private lateinit var image: ImageView
    private lateinit var placeholder: TextView
    private lateinit var progress: ProgressBar
    private lateinit var resultCard: View
    private lateinit var scoreView: TextView
    private lateinit var verdictView: TextView
    private lateinit var warningsView: TextView
    private lateinit var partsView: LinearLayout
    private lateinit var tipsView: TextView

    private val executor = Executors.newSingleThreadExecutor()
    private var cameraFile: File? = null

    private val detector by lazy {
        FaceDetection.getClient(
            FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .setLandmarkMode(FaceDetectorOptions.LANDMARK_MODE_ALL)
                .setContourMode(FaceDetectorOptions.CONTOUR_MODE_ALL)
                .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_ALL)
                .setMinFaceSize(0.15f)
                .build()
        )
    }

    private val takePicture = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok ->
        val f = cameraFile
        if (ok && f != null && f.exists()) {
            runAsync { f.readBytes() }
        }
    }

    private val pickImage = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) runAsync {
            contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: throw IllegalStateException()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        val root = findViewById<View>(R.id.root)
        ViewCompat.setOnApplyWindowInsetsListener(root) { v, insets ->
            val b = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(b.left, b.top, b.right, b.bottom)
            insets
        }

        image = findViewById(R.id.image)
        placeholder = findViewById(R.id.placeholder)
        progress = findViewById(R.id.progress)
        resultCard = findViewById(R.id.resultCard)
        scoreView = findViewById(R.id.score)
        verdictView = findViewById(R.id.verdict)
        warningsView = findViewById(R.id.warnings)
        partsView = findViewById(R.id.parts)
        tipsView = findViewById(R.id.tips)

        cameraFile = savedInstanceState?.getString("cameraFile")?.let { File(it) }

        findViewById<View>(R.id.btnCamera).setOnClickListener { launchCamera() }
        findViewById<View>(R.id.btnGallery).setOnClickListener {
            pickImage.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        findViewById<View>(R.id.btnLink).setOnClickListener { askForLink() }

        // Dozvoluva testiranje/delenje: ako aplikacijata se otvori so link do slika.
        intent?.data?.takeIf { it.scheme == "https" }?.let { loadFromUrl(it.toString()) }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        cameraFile?.let { outState.putString("cameraFile", it.absolutePath) }
    }

    override fun onDestroy() {
        super.onDestroy()
        executor.shutdownNow()
        detector.close()
    }

    private fun launchCamera() {
        try {
            val dir = File(cacheDir, "images").apply { mkdirs() }
            val file = File(dir, "photo_${System.currentTimeMillis()}.jpg")
            cameraFile = file
            val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
            takePicture.launch(uri)
        } catch (e: Exception) {
            toast(R.string.err_no_camera)
        }
    }

    private fun askForLink() {
        val input = EditText(this).apply {
            hint = getString(R.string.link_hint)
            inputType = InputType.TYPE_TEXT_VARIATION_URI
            setSingleLine()
        }
        val pad = (20 * resources.displayMetrics.density).toInt()
        val frame = LinearLayout(this).apply {
            setPadding(pad, pad / 2, pad, 0)
            addView(input, LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)
        }
        AlertDialog.Builder(this)
            .setTitle(R.string.link_title)
            .setView(frame)
            .setPositiveButton(R.string.link_load) { _, _ -> loadFromUrl(input.text.toString()) }
            .setNegativeButton(R.string.cancel, null)
            .show()
    }

    private fun loadFromUrl(raw: String) {
        var url = raw.trim()
        if (url.isEmpty()) return
        if (!url.contains("://")) url = "https://$url"
        if (!url.startsWith("https://", ignoreCase = true)) {
            toast(R.string.err_link)
            return
        }
        runAsync(R.string.err_download) { download(url) }
    }

    /** Preuzima slika (samo https, so limit na golemina i timeout). */
    private fun download(url: String): ByteArray {
        val conn = URL(url).openConnection() as HttpURLConnection
        try {
            conn.connectTimeout = 15_000
            conn.readTimeout = 20_000
            conn.instanceFollowRedirects = true
            conn.setRequestProperty("User-Agent", "MakeupAI/1.0 (Android)")
            conn.setRequestProperty("Accept", "image/*")
            if (conn.responseCode !in 200..299) throw IllegalStateException("HTTP ${conn.responseCode}")
            val type = conn.contentType ?: ""
            if (type.isNotEmpty() && !type.startsWith("image/")) throw IllegalStateException("not an image")
            val limit = 15 * 1024 * 1024
            val out = ByteArrayOutputStream()
            conn.inputStream.use { ins ->
                val buf = ByteArray(16 * 1024)
                while (true) {
                    val n = ins.read(buf)
                    if (n < 0) break
                    out.write(buf, 0, n)
                    if (out.size() > limit) throw IllegalStateException("too large")
                }
            }
            return out.toByteArray()
        } finally {
            conn.disconnect()
        }
    }

    private fun runAsync(errorRes: Int = R.string.err_load, loader: () -> ByteArray) {
        setBusy(true)
        executor.execute {
            try {
                val bitmap = decode(loader())
                runOnUiThread {
                    image.setImageBitmap(bitmap)
                    placeholder.visibility = View.GONE
                }
                val faces = Tasks.await(detector.process(InputImage.fromBitmap(bitmap, 0)))
                val face = faces.maxByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                val report = face?.let { MakeupAnalyzer.analyze(bitmap, it) }
                runOnUiThread {
                    setBusy(false)
                    if (report == null) {
                        resultCard.visibility = View.GONE
                        toast(R.string.err_no_face)
                    } else {
                        show(report)
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    setBusy(false)
                    toast(errorRes)
                }
            }
        }
    }

    /** Dekodira so smaluvanje na golemina i ispravuvanje na rotacija (EXIF). */
    private fun decode(bytes: ByteArray): Bitmap {
        val maxSide = 1600
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
        var sample = 1
        while (max(bounds.outWidth, bounds.outHeight) / (sample * 2) >= maxSide) sample *= 2
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        var bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
            ?: throw IllegalStateException("decode failed")

        val longest = max(bmp.width, bmp.height)
        val m = Matrix()
        if (longest > maxSide) {
            val f = maxSide.toFloat() / longest
            m.postScale(f, f)
        }
        val orientation = try {
            ExifInterface(ByteArrayInputStream(bytes))
                .getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL)
        } catch (e: Exception) {
            ExifInterface.ORIENTATION_NORMAL
        }
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
        }
        if (!m.isIdentity) {
            bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, true)
        }
        return bmp
    }

    private fun setBusy(busy: Boolean) {
        progress.visibility = if (busy) View.VISIBLE else View.GONE
        listOf(R.id.btnCamera, R.id.btnGallery, R.id.btnLink).forEach { findViewById<View>(it).isEnabled = !busy }
        if (busy) resultCard.visibility = View.GONE
    }

    private fun colorFor(score: Int) = ContextCompat.getColor(
        this,
        when {
            score >= 72 -> R.color.good
            score >= 50 -> R.color.ok
            else -> R.color.bad
        }
    )

    private fun show(r: Report) {
        image.setImageBitmap(r.annotated)
        scoreView.text = "${r.score}/100"
        scoreView.setTextColor(colorFor(r.score))
        verdictView.text = r.verdict
        warningsView.visibility = if (r.warnings.isEmpty()) View.GONE else View.VISIBLE
        warningsView.text = r.warnings.joinToString("\n") { "⚠ $it" }

        partsView.removeAllViews()
        val inflater = LayoutInflater.from(this)
        for (p in r.parts) {
            val row = inflater.inflate(R.layout.item_part, partsView, false)
            row.findViewById<TextView>(R.id.partName).text = p.name
            row.findViewById<TextView>(R.id.partScore).text = "${p.score}%"
            row.findViewById<TextView>(R.id.partNote).text = p.note
            row.findViewById<LinearProgressIndicator>(R.id.partBar).apply {
                setIndicatorColor(colorFor(p.score))
                setProgressCompat(p.score, false)
            }
            partsView.addView(row)
        }
        tipsView.text = r.tips.joinToString("\n") { "• $it" }
        resultCard.visibility = View.VISIBLE
    }

    private fun toast(res: Int) = Toast.makeText(this, res, Toast.LENGTH_LONG).show()
}
