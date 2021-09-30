import cv2

import numpy as np


def find_system_ys(org_img, thicken_lines=False):
    img = np.asarray(cv2.cvtColor(org_img * 255, cv2.COLOR_BGR2GRAY), dtype=np.uint8)
    img = 1 - cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 5)/255

    if thicken_lines:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.dilate(img, kernel, iterations=1)

    pxl = img.sum(-1)

    kernel_size = 10
    kernel = np.ones(kernel_size)

    # peaks = scipy.signal.find_peaks(pxl, height=np.max(pxl) / 2)[0]
    pxl_th = pxl.max() / 2
    peaks = np.argwhere(pxl > pxl_th).flatten()
    pxl[:] = 0
    pxl[peaks] = 1

    staff_indicator = np.convolve(pxl, kernel, mode="same")
    staff_indicator[staff_indicator < 1] = 0
    staff_indicator[staff_indicator >= 1] = 1

    diff_values = np.diff(staff_indicator)
    sys_start = np.argwhere(diff_values == 1).flatten()
    sys_end = np.argwhere(diff_values == -1).flatten()

    j = 0

    staffs = []
    for i in range(len(sys_start)):
        s = int(sys_start[i])

        while s >= sys_end[j] and j < len(sys_end):
            j += 1

        e = int(sys_end[j])
        local_peaks = np.argwhere(pxl[s:e + 1] == 1).flatten()
        n_peaks = len(local_peaks)

        # staff has to contain at least 3 peaks and needs to be at least 15 pixel high
        if n_peaks < 3 or local_peaks[-1] - local_peaks[0] < 15:

            staff_indicator[s:e + 1] = 0
            pxl[s:e + 1] = 0
        else:
            pxl[s:e + 1][:local_peaks[0]] = 0
            staff_indicator[s:e + 1][:local_peaks[0]] = 0

            pxl[s:e + 1][local_peaks[-1] + 1:] = 0
            staff_indicator[s:e + 1][local_peaks[-1] + 1:] = 0

            staffs.append((s + local_peaks[0], s + local_peaks[-1]))

    i = 0
    systems = []
    while i + 1 < len(staffs):
        s1 = staffs[i]
        s2 = staffs[i + 1]

        # system has to be at least 30 pixel high
        if s2[1] - s1[0] <= 30:
            i += 1
            continue

        systems.append((s1[0], s2[1]))

        i += 2

    return np.asarray(systems)
