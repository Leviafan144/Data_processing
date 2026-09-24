"""CV-математика: базовые линии, пики, E1/2 и площади.

Вынесено из main() без изменения формул.
Оркестрация (поиск файлов, чтение .mpr, графики, CSV) осталась в main.py.
"""

import logging
import math
import statistics
import decimal
import matplotlib.pyplot as plt

decimal.getcontext().prec = 20


def find_longest_linear_segment(x, y, min_points=10, r2_threshold=decimal.Decimal('0.999')):
    n = len(x)
    if n < min_points:
        return None

    neg_inf = decimal.Decimal('-Infinity')
    zero = decimal.Decimal(0)
    one = decimal.Decimal(1)

    Sx = [zero]
    Sy = [zero]
    Sxx = [zero]
    Sxy = [zero]
    Syy = [zero]
    for xi, yi in zip(x, y):
        Sx.append(Sx[-1] + xi)
        Sy.append(Sy[-1] + yi)
        Sxx.append(Sxx[-1] + xi * xi)
        Sxy.append(Sxy[-1] + xi * yi)
        Syy.append(Syy[-1] + yi * yi)

    for length in range(n, min_points - 1, -1):
        for i in range(0, n - length + 1):
            j = i + length
            sx = Sx[j] - Sx[i]
            sy = Sy[j] - Sy[i]
            sxx = Sxx[j] - Sxx[i]
            sxy = Sxy[j] - Sxy[i]
            syy = Syy[j] - Syy[i]

            denom = length * sxx - sx * sx
            if denom == 0:
                k = zero
                b = sy / length
                r2 = neg_inf
            else:
                k = (length * sxy - sx * sy) / denom
                b = (sy - k * sx) / length
                ss_res = syy - b * sy - k * sxy
                ss_tot = syy - sy * sy / length
                if ss_tot == 0:
                    r2 = one if ss_res == 0 else neg_inf
                else:
                    r2 = one - ss_res / ss_tot

            if r2 >= r2_threshold:
                return (i, j, k, b, r2)
    return None


def simpson_integral(x, y):
    """Legacy, сейчас не используется (площади считает find_square). Оставлен для совместимости."""
    n = len(x) - 1
    if n < 2:
        return decimal.Decimal(0)
    if n % 2 == 1:
        n -= 1
    h = (x[n] - x[0]) / decimal.Decimal(n)
    result = y[0] + y[n]
    for i in range(1, n):
        result += decimal.Decimal(4) * y[i] if i % 2 == 1 else decimal.Decimal(2) * y[i]
    return result * h / decimal.Decimal(3)


def closest_point(i_start, i_finish, loop, point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    s_min = 10 ** 10
    p = loop[i_start]
    for z in range(i_start, i_finish):
        if x1 == x2:
            s = abs(loop[z][1] - y1)
        else:
            s = math.dist(loop[z], (loop[z][0], (y2 - y1) * (loop[z][0] - x1) / (x2 - x1) + y1))
        if s < s_min:
            s_min = s
            p = loop[z]
    return p


def find_y_peak(kind, y_loop, k, b, x_loop):
    y_del_max = 0
    y_peak = y_loop[0]
    for i in range(0, len(y_loop)):
        if kind == "a":
            y_del = y_loop[i] - (k * x_loop[i] + b)
        else:
            y_del = (k * x_loop[i] + b) - y_loop[i]
        if y_del > y_del_max:
            y_peak = y_loop[i]
            y_del_max = y_del

    if y_del_max <= 0:
        logging.warning(f"Peak not found for kind={kind}")

    return y_peak


def draw(panel, x1, x2, x_max, y1, y2, ya, kind, x_peak, y_peak, k, b, y12, p3):
    if kind == "a":
        color = "green"
    else:
        color = "red"
    panel.plot([x1, x2, x_max], [y1, y2, ya], color=color)
    panel.plot([x_peak, x_peak], [y_peak, k * x_peak + b], color=color)
    panel.plot([x1, x2, x_peak, x_max], [y1 + y12, y2 + y12, k * x_peak + b + y12, ya + y12], color=color)
    panel.scatter(x1, y1, color="black", s=7)
    panel.scatter(x2, y2, color="black", s=7)
    panel.scatter(x_peak, y_peak - y12, color="black", s=7)
    panel.scatter(p3[0], p3[1], color="black", s=7)


def find_square(x_loop, y_loop, k, b):
    s = 0
    for i in range(len(x_loop) - 1):
        x1 = x_loop[i]
        y1 = y_loop[i]
        x2 = x_loop[i + 1]
        y2 = y_loop[i + 1]

        y3 = k * x1 + b
        y4 = k * x2 + b

        s += (y1 - y3 + y2 - y4) * (x2 - x1) / 2

    return s


def find_dip(y_loop, smooth_w=15, prom_min=0.05, win=150):
    n = len(y_loop)
    if n < smooth_w + 2:
        return None
    v = [float(t) for t in y_loop]
    w = min(smooth_w, n)
    h = w // 2
    cumsum = [0.0]
    for t in v:
        cumsum.append(cumsum[-1] + t)
    s = []
    for i in range(n):
        lo = max(i - h, 0)
        hi = min(i - h + w, n)
        s.append((cumsum[hi] - cumsum[lo]) / (hi - lo))
    best = None  # (prom, idx)
    for i in range(1, n - 1):
        if s[i] < s[i - 1] and s[i] < s[i + 1]:
            left = max(s[max(0, i - win):i]) if i > 0 else s[i]
            right = max(s[i + 1:min(n, i + 1 + win)])
            prom = min(left, right) - s[i]
            if prom >= prom_min and (best is None or prom > best[0]):
                lo = max(0, i - h)
                hi = min(n, i + h + 1)
                j = min(range(lo, hi), key=lambda t: y_loop[t])
                best = (prom, j)
    return None if best is None else best[1]


def find_dense_blocks(x_loop, y_loop, smooth=11, factor=3.0, gap=5, min_len=20,
                       return_mask=False):
    n = len(x_loop)
    if n < min_len + 2:
        blocks = [(0, n)]
        if not return_mask:
            return blocks
        return blocks, [True] * n
    xs = [float(v) for v in x_loop]
    ys = [float(v) for v in y_loop]
    ex = max(xs) - min(xs)
    ey = max(ys) - min(ys)
    if ex == 0:
        ex = 1.0
    if ey == 0:
        ey = 1.0
    x0, y0 = min(xs), min(ys)
    nx = [(v - x0) / ex for v in xs]
    ny = [(v - y0) / ey for v in ys]
    dist = [math.hypot(nx[i + 1] - nx[i], ny[i + 1] - ny[i]) for i in range(n - 1)]
    med = statistics.median(dist)
    if med == 0:
        blocks = [(0, n)]
        if not return_mask:
            return blocks
        return blocks, [True] * n
    w = max(1, min(smooth, len(dist)))
    half = w // 2
    cumsum = [0.0]
    for d in dist:
        cumsum.append(cumsum[-1] + d)
    sm = []
    for i in range(len(dist)):
        lo = max(i - half, 0)
        hi = min(i - half + w, len(dist))
        sm.append((cumsum[hi] - cumsum[lo]) / (hi - lo))
    thr = med * factor
    sparse = [i for i, d in enumerate(sm) if d > thr]
    bends = []
    if sparse:
        s = [sparse[0]]
        for i in sparse[1:]:
            if i - s[-1] <= gap:
                s.append(i)
            else:
                bends.append((s[0], s[-1]))
                s = [i]
        bends.append((s[0], s[-1]))
    blocks = []
    prev = 0
    for b0, b1 in bends:
        blocks.append((prev, b0))
        prev = b1 + 1
    blocks.append((prev, n))
    blocks = [(a, b) for a, b in blocks if b - a >= min_len]
    if not return_mask:
        return blocks
    mask = [False] * n
    for a, b in blocks:
        for i in range(a, min(b, n)):
            mask[i] = True
    return blocks, mask


def _lsq(x, y):
    m = len(x)
    if m < 2:
        return None
    zero = decimal.Decimal(0)
    sx = sum(x, zero)
    sy = sum(y, zero)
    sxx = sum(v * v for v in x)
    sxy = sum(xx * yy for xx, yy in zip(x, y))
    denom = m * sxx - sx * sx
    if denom == 0:
        return None
    k = (m * sxy - sx * sy) / denom
    b = (sy - k * sx) / m
    return (k, b)


def find_baseline(x, y, lo, hi, r2_threshold=decimal.Decimal('0.999'), min_len=30):
    n = len(x)
    zero = decimal.Decimal(0)
    one = decimal.Decimal(1)

    lo = max(0, lo)
    hi = min(n, hi)
    if hi - lo < min_len:
        return None

    Sx = [zero]
    Sy = [zero]
    Sxx = [zero]
    Sxy = [zero]
    Syy = [zero]
    for xi, yi in zip(x, y):
        Sx.append(Sx[-1] + xi)
        Sy.append(Sy[-1] + yi)
        Sxx.append(Sxx[-1] + xi * xi)
        Sxy.append(Sxy[-1] + xi * yi)
        Syy.append(Syy[-1] + yi * yi)

    best = None  # ((abs_slope, -length, i), i, j, k, b, r2)
    for length in range(hi - lo, min_len - 1, -1):
        for i in range(lo, hi - length + 1):
            j = i + length
            sx = Sx[j] - Sx[i]
            sy = Sy[j] - Sy[i]
            sxx = Sxx[j] - Sxx[i]
            sxy = Sxy[j] - Sxy[i]
            syy = Syy[j] - Syy[i]

            denom = length * sxx - sx * sx
            if denom == 0:
                continue
            k = (length * sxy - sx * sy) / denom
            b = (sy - k * sx) / length
            ss_res = syy - b * sy - k * sxy
            ss_tot = syy - sy * sy / length
            if ss_tot == 0:
                if ss_res != 0:
                    continue
                r2 = one
            else:
                r2 = one - ss_res / ss_tot

            if r2 >= r2_threshold:
                key = (abs(float(k)), -length, i)
                if best is None or key < best[0]:
                    best = (key, i, j, k, b, r2)
    if best is None:
        return None
    _, i, j, k, b, r2 = best
    return (i, j, k, b, r2)


def _count_branch(loop, x_loop, y_loop, x_end, del_x, del_y, axe, loop_num, kind):
    try:
        if kind == "a":
            peak_idx = y_loop.index(max(y_loop))
            dip_idx = None
            wall = max(10, peak_idx // 20)
            seg = find_baseline(x_loop, y_loop, 0, peak_idx - wall)
            if seg is None:
                # fallback: старый глобальный поиск
                logging.warning("Baseline not found for kind=a, fallback to full range")
                seg = find_longest_linear_segment(x_loop[:peak_idx], y_loop[:peak_idx])
        else:
            # катод: верхняя часть первого плотного блока (как исходная
            # программа: напр. [34,76) при провале на E 0.759), пик — провал
            dip_idx = find_dip(y_loop)
            if dip_idx is None:
                dip_idx = y_loop.index(min(y_loop))
            peak_idx = dip_idx
            blocks, _ = find_dense_blocks(x_loop, y_loop, return_mask=True)
            seg = None
            if blocks:
                a, b = max(blocks[0][0], 0), min(blocks[0][1], dip_idx)
                if b - a >= 10:
                    s = find_longest_linear_segment(x_loop[a:b], y_loop[a:b])
                    if s is not None:
                        seg = (s[0] + a, s[1] + a, s[2], s[3], s[4])
            if seg is None:
                logging.warning("Cathodic background not found")
                return (0, 0, 0, 0, 0, 0, 0, 0)
        start, end, k, b, r2 = seg
    except (TypeError, ValueError, AttributeError):
        logging.error("Doesn't find the longest linear segment")
        return (0, 0, 0, 0, 0, 0, 0, 0)

    x1, y1, x2, y2 = loop[start][0], loop[start][1], loop[end - 1][0], loop[end - 1][1]
    ya = k * x_end + b

    if kind == "c":
        y_peak = y_loop[dip_idx]
        x_peak = loop[dip_idx][0]
    else:
        y_peak = find_y_peak(kind, y_loop, k, b, x_loop)
        x_peak = loop[y_loop.index(y_peak)][0]

    y12 = (y_peak - (k * x_peak + b)) / 2

    p3 = closest_point(x_loop.index(x2), y_loop.index(y_peak), loop, (x1, y1 + y12), (x2, y2 + y12))

    e12 = p3[0] / del_x
    ep = x_peak / del_x
    yp = 2 * y12 / del_y
    if axe is not None:
        draw(axe, x1, x2, x_end, y1, y2, ya, kind, x_peak, y_peak, k, b, y12, p3)
        if loop_num == 2:
            draw(plt, x1, x2, x_end, y1, y2, ya, kind, x_peak, y_peak, k, b, y12, p3)

    return (e12, ep, yp, k, b, y_peak, start, end)


def analyze_loop(loop, del_x, del_y, x_max, axe=None, loop_num=0):
    x_loop = [loop[z][0] for z in range(len(loop))]
    y_loop = [loop[z][1] for z in range(len(loop))]
    x_max_idx = x_loop.index(x_max)

    x_loop_a = x_loop[:x_max_idx + 1]
    y_loop_a = y_loop[:x_max_idx + 1]
    loopa = loop[:x_max_idx + 1]

    x_loop_c = x_loop[x_max_idx + 1:]
    y_loop_c = y_loop[x_max_idx + 1:]
    loopc = loop[x_max_idx + 1:]

    x_max_a = x_loop_a[-1]
    x_min_c = x_loop_c[-1] if x_loop_c else x_max

    e12a, epa, ypa, ka, ba, y_peak_a, start_a, end_a = _count_branch(
        loopa, x_loop_a, y_loop_a, x_max_a, del_x, del_y, axe, loop_num, "a")

    e12c, epc, ypc, kc, bc, y_peak_c, start_c, end_c = _count_branch(
        loopc, x_loop_c, y_loop_c, x_min_c, del_x, del_y, axe, loop_num, "c")

    anodic_ok = not (ka == 0 and ba == 0 and start_a == 0 and end_a == 0)
    cathodic_ok = not (kc == 0 and bc == 0 and start_c == 0 and end_c == 0)

    if anodic_ok:
        try:
            c_end = y_loop_c.index(y_peak_c)
        except ValueError:
            c_end = len(loopc)
        p4 = closest_point(0, c_end, loopc,
                           (x_loop[start_a], y_loop[start_a]), (x_loop[end_a], y_loop[end_a]))
        intersection_c_idx = loopc.index(p4)
        area_anodic_a = find_square(x_loop_a[end_a:], y_loop_a[end_a:], ka, ba)
        area_cathodic_a = find_square(x_loop_c[:intersection_c_idx + 1], y_loop_c[:intersection_c_idx + 1], ka, ba)
        spa = area_anodic_a + area_cathodic_a
    else:
        e12a, epa, ypa, spa = 0, 0, 0, 0

    if cathodic_ok:
        p5 = closest_point(0, y_loop_a.index(y_peak_a), loopa,
                           (x_loop_c[start_c], y_loop_c[start_c]),
                           (x_loop_c[end_c - 1], y_loop_c[end_c - 1]))
        intersection_a_idx = loopa.index(p5)
        try:
            peak_c_idx = y_loop_c.index(y_peak_c)
        except ValueError:
            peak_c_idx = 0
        if end_c <= peak_c_idx:
            lo_c, hi_c = end_c, peak_c_idx
        else:
            lo_c, hi_c = peak_c_idx, end_c
        area_cathodic_c = find_square(x_loop_c[lo_c:hi_c], y_loop_c[lo_c:hi_c], kc, bc)
        area_anodic_c = find_square(x_loop_a[:intersection_a_idx + 1], y_loop_a[:intersection_a_idx + 1], kc, bc)
        spc = area_cathodic_c + area_anodic_c
    else:
        e12c, epc, ypc, spc = 0, 0, 0, 0

    return (e12a, epa, ypa, spa, e12c, epc, ypc, spc)
