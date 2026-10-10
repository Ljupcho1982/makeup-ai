package com.ilche.makeupai

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import com.google.mlkit.vision.face.Face
import com.google.mlkit.vision.face.FaceContour
import com.google.mlkit.vision.face.FaceLandmark
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

data class Part(val name: String, val score: Int, val note: String)

data class Report(
    val score: Int,
    val verdict: String,
    val parts: List<Part>,
    val tips: List<String>,
    val warnings: List<String>,
    val annotated: Bitmap,
)

/**
 * Evristichka ocenka na shminka vrz osnova na boja (CIE Lab) vo regioni na liceto
 * shto gi dava ML Kit (usni, ochi, vegji, obrazi). Sekogash se sporeduva so tenot na istata slika,
 * pa osvetluvanjeto vlijae pomalku. Ova e procenka, ne "vistinska" nauchna merka.
 */
object MakeupAnalyzer {

    private class Lab(val l: Double, val a: Double, val b: Double) {
        fun dist(o: Lab) = sqrt((l - o.l).pow(2) + (a - o.a).pow(2) + (b - o.b).pow(2))
    }

    private fun toLab(c: Int): Lab {
        fun lin(v: Int): Double {
            val s = v / 255.0
            return if (s <= 0.04045) s / 12.92 else ((s + 0.055) / 1.055).pow(2.4)
        }
        val r = lin(Color.red(c))
        val g = lin(Color.green(c))
        val bl = lin(Color.blue(c))
        val x = (0.4124 * r + 0.3576 * g + 0.1805 * bl) / 0.95047
        val y = 0.2126 * r + 0.7152 * g + 0.0722 * bl
        val z = (0.0193 * r + 0.1192 * g + 0.9505 * bl) / 1.08883
        fun f(t: Double) = if (t > 0.008856) t.pow(1.0 / 3.0) else 7.787 * t + 16.0 / 116.0
        val fx = f(x)
        val fy = f(y)
        val fz = f(z)
        return Lab(116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz))
    }

    private fun median(v: List<Lab>): Lab? {
        if (v.isEmpty()) return null
        fun m(sel: (Lab) -> Double): Double = v.map(sel).sorted()[v.size / 2]
        return Lab(m { it.l }, m { it.a }, m { it.b })
    }

    private fun stdL(v: List<Lab>): Double {
        if (v.size < 2) return 0.0
        val mean = v.sumOf { it.l } / v.size
        return sqrt(v.sumOf { (it.l - mean).pow(2) } / v.size)
    }

    private fun inside(x: Float, y: Float, p: List<PointF>): Boolean {
        var c = false
        var j = p.size - 1
        for (i in p.indices) {
            if ((p[i].y > y) != (p[j].y > y) &&
                x < (p[j].x - p[i].x) * (y - p[i].y) / (p[j].y - p[i].y) + p[i].x
            ) c = !c
            j = i
        }
        return c
    }

    /** Gi podreduva tochkite okolu centroidot za da dobieme prost (nesamopreseken) poligon. */
    /** Polygon od dva "rabovi": prviot sortiran po x, vtoriot po x vo obratna nasoka. */
    private fun band(top: List<PointF>, bottom: List<PointF>): List<PointF> =
        top.sortedBy { it.x } + bottom.sortedByDescending { it.x }

    private fun sortAround(pts: List<PointF>): List<PointF> {
        if (pts.size < 3) return pts
        val cx = pts.map { it.x }.average().toFloat()
        val cy = pts.map { it.y }.average().toFloat()
        return pts.sortedBy { atan2((it.y - cy).toDouble(), (it.x - cx).toDouble()) }
    }

    private class Sampler(bmp: Bitmap) {
        val w = bmp.width
        val h = bmp.height
        val px = IntArray(w * h).also { bmp.getPixels(it, 0, w, 0, 0, w, h) }

        fun patch(cx: Float, cy: Float, r: Float): List<Lab> {
            val out = ArrayList<Lab>()
            val x0 = max(0, (cx - r).toInt())
            val x1 = min(w - 1, (cx + r).toInt())
            val y0 = max(0, (cy - r).toInt())
            val y1 = min(h - 1, (cy + r).toInt())
            for (y in y0..y1) for (x in x0..x1) {
                val dx = x - cx
                val dy = y - cy
                if (dx * dx + dy * dy <= r * r) out.add(toLab(px[y * w + x]))
            }
            return out
        }

        fun poly(outer: List<PointF>, hole: List<PointF>? = null): List<Lab> {
            if (outer.size < 3) return emptyList()
            val out = ArrayList<Lab>()
            val x0 = max(0, outer.minOf { it.x }.toInt())
            val x1 = min(w - 1, outer.maxOf { it.x }.toInt())
            val y0 = max(0, outer.minOf { it.y }.toInt())
            val y1 = min(h - 1, outer.maxOf { it.y }.toInt())
            for (y in y0..y1) for (x in x0..x1) {
                val fx = x + 0.5f
                val fy = y + 0.5f
                if (!inside(fx, fy, outer)) continue
                if (hole != null && hole.size >= 3 && inside(fx, fy, hole)) continue
                out.add(toLab(px[y * w + x]))
            }
            return out
        }
    }

    private fun scale(v: Double, lo: Double, hi: Double, sLo: Double, sHi: Double): Double {
        val t = ((v - lo) / (hi - lo)).coerceIn(0.0, 1.0)
        return sLo + t * (sHi - sLo)
    }

    private fun pick(score: Int, low: String, mid: String, high: String) =
        when {
            score < 45 -> low
            score < 72 -> mid
            else -> high
        }

    fun analyze(src: Bitmap, face: Face): Report? {
        val box = face.boundingBox
        val fw = box.width().toFloat()
        val fh = box.height().toFloat()
        val s = Sampler(src)

        fun contour(type: Int) = face.getContour(type)?.points?.map { PointF(it.x, it.y) } ?: emptyList()

        val upperLipTop = contour(FaceContour.UPPER_LIP_TOP)
        val lowerLipBottom = contour(FaceContour.LOWER_LIP_BOTTOM)
        val upperLipBottom = contour(FaceContour.UPPER_LIP_BOTTOM)
        val lowerLipTop = contour(FaceContour.LOWER_LIP_TOP)
        val leftEye = contour(FaceContour.LEFT_EYE)
        val rightEye = contour(FaceContour.RIGHT_EYE)
        val browPts = listOf(
            band(contour(FaceContour.LEFT_EYEBROW_TOP), contour(FaceContour.LEFT_EYEBROW_BOTTOM)),
            band(contour(FaceContour.RIGHT_EYEBROW_TOP), contour(FaceContour.RIGHT_EYEBROW_BOTTOM)),
        )
        val faceOutline = contour(FaceContour.FACE)
        if (upperLipTop.isEmpty() || lowerLipBottom.isEmpty() || leftEye.isEmpty() || rightEye.isEmpty() ||
            browPts.any { it.isEmpty() } || faceOutline.isEmpty()
        ) return null

        // ---- Referentna boja na ten (celo, obrazi, brada) ----
        val allBrowTop = contour(FaceContour.LEFT_EYEBROW_TOP) + contour(FaceContour.RIGHT_EYEBROW_TOP)
        val browTopY = allBrowTop.minOf { it.y }
        val browCx = allBrowTop.map { it.x }.average().toFloat()
        val foreheadY = ((browTopY + box.top) / 2f).coerceAtMost(browTopY - 0.04f * fh)
        val chinBottom = faceOutline.maxOf { it.y }
        val lipBottom = lowerLipBottom.maxOf { it.y }
        val chinY = lipBottom + 0.45f * (chinBottom - lipBottom)
        val mouthCx = (upperLipTop + lowerLipBottom).map { it.x }.average().toFloat()

        val lc = face.getLandmark(FaceLandmark.LEFT_CHEEK)?.position
        val rc = face.getLandmark(FaceLandmark.RIGHT_CHEEK)?.position
        val cheekL = lc ?: PointF(box.centerX() - 0.24f * fw, box.centerY() + 0.12f * fh)
        val cheekR = rc ?: PointF(box.centerX() + 0.24f * fw, box.centerY() + 0.12f * fh)

        val rBig = 0.06f * fw
        val pForehead = s.patch(browCx, foreheadY, rBig)
        val pChin = s.patch(mouthCx, chinY, 0.05f * fw)
        val pCheekL = s.patch(cheekL.x, cheekL.y, rBig)
        val pCheekR = s.patch(cheekR.x, cheekR.y, rBig)
        val mForehead = median(pForehead)
        val mChin = median(pChin)
        val mCheekL = median(pCheekL)
        val mCheekR = median(pCheekR)
        if (mForehead == null || mChin == null || mCheekL == null || mCheekR == null) return null

        val skin = median(pForehead + pChin + pCheekL + pCheekR)!!

        // ---- Usni ----
        val lipOuter = sortAround(upperLipTop + lowerLipBottom)
        val lipHole = sortAround(upperLipBottom + lowerLipTop)
        val lipLab = median(s.poly(lipOuter, lipHole)) ?: return null
        val lipDelta = lipLab.dist(skin)
        val lipScore = scale(lipDelta, 6.0, 30.0, 25.0, 100.0).toInt()

        // ---- Ochi: kapak (senka) + linija na trepkite ----
        val eyeRegions = ArrayList<List<PointF>>()
        val lidLabs = ArrayList<Lab>()
        val lashLabs = ArrayList<Lab>()
        for (eye in listOf(leftEye, rightEye)) {
            val cy = eye.map { it.y }.average().toFloat()
            val eyeH = eye.maxOf { it.y } - eye.minOf { it.y }
            val dy = max(0.9f * eyeH, 0.05f * fw)
            val upper = eye.filter { it.y <= cy }
            if (upper.size < 3) continue
            val lid = band(upper.map { PointF(it.x, it.y - dy) }, upper)
            eyeRegions.add(lid)
            lidLabs += s.poly(lid)
            upper.forEach { lashLabs += s.patch(it.x, it.y, max(1.5f, 0.008f * fw)) }
        }
        if (lashLabs.isEmpty()) return null
        val lidMed = median(lidLabs) ?: return null
        val lidDelta = lidMed.dist(skin)
        val lashDark = skin.l - lashLabs.map { it.l }.sorted().let { it[(it.size * 0.3).toInt()] }
        val lidScore = scale(lidDelta, 4.0, 28.0, 25.0, 100.0)
        val lashScore = scale(lashDark, 20.0, 55.0, 25.0, 100.0)
        val eyeScore = (0.6 * lidScore + 0.4 * lashScore).toInt()

        // ---- Vegji ----
        val browRegions = browPts
        val browLab = median(browRegions.flatMap { s.poly(it) }) ?: return null
        val browDelta = skin.l - browLab.l
        val browScore = scale(browDelta, 8.0, 38.0, 25.0, 100.0).toInt()

        // ---- Ten (izramnetost) ----
        val sigma = listOf(pForehead, pChin, pCheekL, pCheekR).map { stdL(it) }.average()
        val patchDelta = max(
            sqrt((mForehead.a - mChin.a).pow(2) + (mForehead.b - mChin.b).pow(2)),
            sqrt((mCheekL.a - mCheekR.a).pow(2) + (mCheekL.b - mCheekR.b).pow(2)),
        )
        val skinScore = (100 - (sigma - 4.0).coerceAtLeast(0.0) * 5.0 - (patchDelta - 5.0).coerceAtLeast(0.0) * 3.0)
            .coerceIn(30.0, 100.0).toInt()

        // ---- Rumenilo ----
        val cheekA = (mCheekL.a + mCheekR.a) / 2
        val baseA = (mForehead.a + mChin.a) / 2
        val blushDelta = cheekA - baseA
        var blushScore = scale(blushDelta, 1.0, 6.5, 35.0, 100.0)
        if (blushDelta > 12) blushScore -= (blushDelta - 12) * 4
        val blush = blushScore.coerceIn(30.0, 100.0).toInt()

        // ---- Balans (eden akcent: usni ILI ochi) ----
        val balance: Int
        val balanceNote: String
        when {
            lipScore > 85 && eyeScore > 85 -> {
                balance = 70
                balanceNote = "I očite i usnite se mnogu istaknati. Obično izgleda najdobro so eden glaven akcent."
            }
            lipScore < 55 && eyeScore < 55 -> {
                balance = 60
                balanceNote = "Nema jasen akcent — nitu oči, nitu usni ne se istaknati."
            }
            else -> {
                balance = 92
                balanceNote = "Ima jasen akcent i izgledot e izbalansiran."
            }
        }

        val parts = listOf(
            Part("Ten", skinScore, pick(skinScore, "Tenot deluva neramnomeren ili so vidlivi razliki vo boja.", "Tenot e prilično ramnomeren.", "Tenot e mnogu ramnomeren i izramnet.")),
            Part("Usni", lipScore, pick(lipScore, "Usnite se vo boja bliska do tenot — rečisi bez boja.", "Usnite imaat lesna boja.", "Usnite se jasno definirani so boja.")),
            Part("Oči", eyeScore, pick(eyeScore, "Očite se bez definicija (nema senka/linija).", "Ima lesna definicija na očite.", "Očite se dobro istaknati (senka i/ili linija).")),
            Part("Vegji", browScore, pick(browScore, "Vegjite se bledi/slabo definirani.", "Vegjite se umereno definirani.", "Vegjite se jasno definirani.")),
            Part("Rumenilo", blush, pick(blush, "Ne se zabeležuva rumenilo na obrazite.", "Ima lesna svežina na obrazite.", if (blushDelta > 12) "Rumeniloto e dosta intenzivno." else "Rumeniloto na obrazite e ubavo istaknato.")),
            Part("Balans", balance, balanceNote),
        )

        // ---- Vkupno ----
        val makeupLevel = (lipScore + eyeScore + browScore + blush) / 4.0
        var overall = (0.25 * skinScore + 0.20 * lipScore + 0.20 * eyeScore + 0.15 * browScore + 0.10 * blush + 0.10 * balance)
        val bare = makeupLevel < 45
        if (bare) overall = min(overall, 45.0)
        val total = overall.toInt().coerceIn(0, 100)
        val verdict = when {
            bare -> "Rečisi bez šminka"
            total >= 82 -> "Odlično našminkana"
            total >= 68 -> "Ubavo našminkana"
            total >= 55 -> "Umerena šminka, može podobro"
            else -> "Šminkata e slaba ili neramnomerna"
        }

        // ---- Saveti ----
        val tips = ArrayList<String>()
        if (skinScore < 65) tips.add("Ten: izramni so korektor/pudra i dobro blendiraj, osobeno okolu nosot i bradata.")
        if (lipScore < 55) tips.add("Usni: probaj karmin ili sjaj vo ton što se slaga so tenot.")
        if (eyeScore < 55) tips.add("Oči: dodaj senka vo neutralni tonovi i tenka linija/maskara za definicija.")
        if (browScore < 55) tips.add("Vegji: popolni gi so moliv ili gel vo boja bliska do kosata.")
        if (blush < 55) tips.add("Rumenilo: malku rumenilo na jagodicite dava svežina.")
        if (blushDelta > 12) tips.add("Rumenilo: blendiraj go povekje za ponežen premin.")
        if (balance < 75 && lipScore > 85 && eyeScore > 85) tips.add("Izberi eden akcent: intenzivni usni so nežni oči, ili obratno.")
        if (tips.isEmpty()) tips.add("Odličen rezultat. Šminkata e izbalansirana i dobro izvršena.")

        // ---- Preduprezduvanja za kvalitet na slikata ----
        val warnings = ArrayList<String>()
        if (fw < 150) warnings.add("Liceto e malo na slikata — rezultatot može da e neprecizen. Približi se.")
        if (abs(face.headEulerAngleY) > 25 || abs(face.headEulerAngleZ) > 20)
            warnings.add("Liceto e svrteno/nakriveno — probaj frontalna slika.")
        if (skin.l < 30) warnings.add("Slikata e premračna — probaj podobro osvetluvanje.")
        if (skin.l > 92) warnings.add("Slikata e preosvetlena — probaj pomalku direktna svetlina.")
        val lo = face.leftEyeOpenProbability
        val ro = face.rightEyeOpenProbability
        if ((lo != null && lo < 0.3f) || (ro != null && ro < 0.3f))
            warnings.add("Očite izgledaat zatvoreni — ocenkata za oči može da e neprecizna.")

        // ---- Anotirana slika (koi regioni se analizirani) ----
        val annotated = src.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(annotated)
        val stroke = max(2f, fw * 0.006f)
        fun draw(poly: List<PointF>, color: Int) {
            if (poly.size < 3) return
            val path = Path()
            path.moveTo(poly[0].x, poly[0].y)
            for (i in 1 until poly.size) path.lineTo(poly[i].x, poly[i].y)
            path.close()
            canvas.drawPath(path, Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.STROKE
                strokeWidth = stroke
                this.color = color
            })
        }
        draw(lipOuter, Color.rgb(255, 64, 129))
        eyeRegions.forEach { draw(it, Color.rgb(64, 196, 255)) }
        browRegions.forEach { draw(it, Color.rgb(255, 215, 64)) }
        val circlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = stroke
            color = Color.rgb(178, 255, 89)
        }
        canvas.drawCircle(browCx, foreheadY, rBig, circlePaint)
        canvas.drawCircle(mouthCx, chinY, 0.05f * fw, circlePaint)
        canvas.drawCircle(cheekL.x, cheekL.y, rBig, circlePaint)
        canvas.drawCircle(cheekR.x, cheekR.y, rBig, circlePaint)

        return Report(total, verdict, parts, tips, warnings, annotated)
    }
}
