from scipy.signal import find_peaks, peak_widths, peak_prominences
import matplotlib.pyplot as plt
import time, re, math, argparse, os
import numpy as np
from collections import deque


# read data file
def readFile(filename):

    data = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            try:
                row = list(map(int, re.split(r'\s{1,}', line.strip())))
                mVal = max(row[1:54])
                row.append(mVal)
                data.append(row)
            except ValueError as e:
                print(e)
                print(f"{line}")

    arr = np.array([row[-1] for row in data])
    return data, arr



# adjust timestamp with rollover count
# returns updated timestamp and cracks (a list of huge timestamp diff interval)
def updateTimestamp(d):

    # Extract timestamp column
    timestamp = d[:, 0]

    # Calculate rollover count
    diff = np.diff(timestamp)
    rollover_count = np.zeros(len(timestamp), dtype=int)
    rollover_count[1:] = np.cumsum(diff < 0)

    # # Convert timestamps to time values
    t = timestamp + 65536 * rollover_count

    cracks = []
    m = len(t)
    for i in range(1, m):
        new_diff = t[i] - t[i-1]
        if new_diff > 300:
            cracks.append((i-1, i))
        # t = np.concatenate((t[:i], new), axis = None)

    # print(f"The length of t = {len(t)}")
    print(f"Cracks = {cracks}")
    return t, cracks



# save point information to {filename}_out.txt
# (ith touch, start_index, end_index, the number of dataframes within the touch interval)
# marked missing points as (start_index, end_index) = (-1, -1)
def savePointInfo(pointsInfo, output):
    print(f"\nSaving info into {output}...")
    missing = 0 
    with open(output, 'w+') as f:
        for i in range(0, len(pointsInfo)):
            s, e = pointsInfo[i]
            if (s, e) == (-1, -1):
                missing += 1
            f.write(f"{i} {s} {e} {e-s+1}\n")
    print(f"--> {missing} missing points were included, as a form of (-1, -1).")
    print(f"--> Saved {len(pointsInfo)} contacts in total.")
    return missing



# detect contact and release points
# returns np array of (start_index, end_index, peak_index, peak_width, peak_height)
def pointDetection(max_data, n, isPoint) -> list:

    # size for convolution (may need to change depending on the data type)
    # relative height of the width of each peak
    if isPoint:             # works for 1mm points
        size = 40               
        rel_height = 0.2
        dist = 80        
    else:                   # works for lines (edge to edge)
        size = 200          # 200, 0.7
        rel_height = 0.7
        dist = 200

    # convolve the data
    # may need to change window size depending on the data type (size below works for 1mm, 10833 points)
    window = np.ones(int(size))/float(size)
    x = np.convolve(max_data, window, 'same')

    # find all peaks including noises (to use for 1st filter)
    peaks0, _ = find_peaks(x, distance=80)
    print(f"The number of all peaks including noises = {len(peaks0)}")
    pro = peak_prominences(x, peaks0)

    # find about n peaks with calculated prominence
    peaks, _ = find_peaks(x, distance=dist, prominence=np.percentile(pro[0], 100*(1-n/len(peaks0))))
    print(f"The number of peaks after 1st filtering = {len(peaks)}")

    # find the width of each peak (to use for 2nd filter)
    wdt_results = peak_widths(x, peaks, rel_height=rel_height)

    # create pointsInfo, array of (start_inded, end_index, peak_index, peak_width, peak_height)
    pointsInfo = []
    for i in range(len(wdt_results[0])):
        peak_height = x[peaks[i]]
        width_height = wdt_results[1][i]
        start = math.ceil(wdt_results[2][i])
        width = round(wdt_results[0][i])

        # the issue of using prominence, it looks for n number of prominent points
        # therefore when the data was missing some points, this filter is necessary
        # filter out not-real points (where convolution size matters)
        if peak_height - width_height < 1:
            # print(f"Filtered... {i}, {peak_height} {width_height}")
            # print(f"start = {start}, end = {start+width}")
            continue
        pointsInfo.append((start, start+width, peaks[i], width, height))
    
    print(f"The number of peaks after 2nd filtering = {len(pointsInfo)}")

    pointsInfo = np.array(pointsInfo, dtype=int)

    # plot the data with its detected (and filtered) peaks
    # plt.plot(max_data)
    # plt.plot(x)
    # plt.plot(pointsInfo[:, 2], x[pointsInfo[:, 2]], "x")
    # plt.hlines(*wdt_results[1:], color="C2")
    # plt.show()

    # return array of (start_index, end_index)
    return pointsInfo[:, :2]



# not smart enough to know which edge we are missing points
def validateRows(t, pointsInfo, width, height):

    middle_rows = deque()

    s, e = pointsInfo[0]
    prev = t[e]
    temp = [0]
    for i in range(1, len(pointsInfo)):
        s, e = pointsInfo[i]
        temp.append(t[s]-prev-1)
        prev = t[e]

    # find a middle valid row
    start = (height//2)*width
    found, j = False, None
    while not found:

        # change point?
        if temp[start] - temp[start+1] > 0.3 * temp[start]:
            # do they have valid points of width?
            j = 1
            while abs(temp[start+j] - temp[start+j+1]) < 0.3 * max(temp[start+j], temp[start+j+1]):
                j += 1
            if j == width-1:
                found = True
        else:
            start = start + (j+1) if j else start + 1


    middle_rows.append(pointsInfo[start:start+width].tolist())

    # to the top
    t_idx = -1
    for i in range(start-1, -1, -width):

        if i - width + 1 < 0:           # if the row has less number of values than width
            t_idx = i
        elif i - width + 1 != 0:        # if the row has width number of values to validate
            maxV = temp[i-width+1]
            mean = maxV
            # check if this entire row's max value is at index 0
            for j in range(i+2-width, i+1):
                if maxV < temp[j]:
                    t_idx = i
                    break
                mean += temp[j]
        
            # check the max value's significance
            mean /= width
            if t_idx != -1 and maxV < mean*1.3:
                t_idx = i

        if t_idx != -1:
            # print(f"{t_idx//width}th row: suspicious of missing points")
            # print(temp[max(0, i-width+1):i+1])
            # print(pointsInfo[max(0, i-width+1):i+1])
            break
        else:
            # print(temp[i-width+1:i+1])
            middle_rows.appendleft(pointsInfo[i-width+1:i+1].tolist())


    # to the bottom
    b_idx = -1
    for i in range(start+width, len(temp), width):


        if i + width < len(temp):    # if the row has width number of values to validate    
            maxV = temp[i]
            mean = maxV
            # check if this entire row's max value is at index 0
            for j in range(i+1, i+width):
                if maxV < temp[j]:
                    b_idx = i
                    break
                mean += temp[j]

            # check the max value's significance
            mean /= width
            if b_idx != -1 and maxV < mean*1.3:
                print(maxV)
                b_idx = i

        elif i + width != len(temp):       # if the row has less number of values than width
            b_idx = i

        if b_idx != -1:
            # print(f"{b_idx//width}th row: suspicious of missing points")
            # print(temp[i:min(i+width, len(temp))])
            # print(pointsInfo[i:min(i+width, len(temp))])
            break
        else:
            # print(temp[i:i+width])
            middle_rows.append(pointsInfo[i:i+width].tolist())

    print(f"t_idx = {t_idx}, b_idx = {b_idx}")
    # print(f"The number of normal rows in the middle = {len(middle_rows)}")
    return t_idx, b_idx, middle_rows



# calculate normal row's average touch and non touch points
def summarizeRow(t, row, width):
    prev_start_ts, prev_end_ts = t[row[0][0]], t[row[0][1]]
    average_interval_of_points, average_interval_btw_points = prev_end_ts - prev_start_ts, 0

    for j in range(1, width):
        s, e = row[j]
        average_interval_of_points += t[e] - t[s]
        average_interval_btw_points += t[s] - prev_end_ts
        prev_end_ts = t[e]

    average_interval_btw_points /= (width-1)
    average_interval_of_points /= (width)

    # print(round(average_interval_of_points, 2), round(average_interval_btw_points, 2))
    return average_interval_of_points, average_interval_btw_points



def summarizeRow2(ts_row, width):
    prev_start_ts, prev_end_ts = ts_row[0][0], ts_row[0][1]
    average_interval_of_points, average_interval_btw_points = prev_end_ts - prev_start_ts, 0

    for j in range(1, width):
        average_interval_of_points += ts_row[j][1] - ts_row[j][0]
        average_interval_btw_points += ts_row[j][0] - prev_end_ts
        prev_end_ts = ts_row[j][1]

    average_interval_btw_points /= (width-1)
    average_interval_of_points /= (width)

    # print(round(average_interval_of_points, 2), round(average_interval_btw_points, 2))
    return average_interval_of_points, average_interval_btw_points



# checks where a given row doesn't have a point at an expected timestamp
# from the end of the line (looking for a valid end point)
def validateColumns_rvs(t, pointsInfo, k, saved_normal, next_start_ts, thrd):
    points_itv, btw_points_itv, moving_itv = saved_normal
    touch_itv = points_itv + btw_points_itv
    new_rows = []

    j = k
    cnt = missing = 0

    while cnt  <= k//width:

        pointsInfo_idx = [(-1, -1) for _ in range(width)]
        valid_ts = [(-1, -1) for _ in range(width)]

        expected_end = next_start_ts - moving_itv
        pos = width - 1
        s, e = pointsInfo[j]

        # print(f"{cnt} iteration: j = {j}, pos = {pos}")
        # print(pointsInfo[j-width:j+1])

        # dealing with end of each line(row)
        if abs(expected_end - t[e]) > thrd: #t[e] * 0.001:
            print(f"A point is missing at ({k//width-cnt}, {pos})")
            print(f"[INFO] expected_end = {round(expected_end, 2)}, real_end = {t[e]}")
            # print(pointsInfo[j])
            prev_end = expected_end
            valid_ts[pos] = (expected_end-points_itv, expected_end) #(expected_end-touch_itv, expected_end)
            missing += 1
        else:
            prev_end = t[e]
            j -= 1
            pointsInfo_idx[pos] = s, e
            valid_ts[pos] = (t[s], t[e]) 
        pos -= 1

        # valiadting the rest of each line(row) with intervals between two points
        while pos >= 0 and j >= 0:

            s, e = pointsInfo[j]

            expected_end = prev_end - touch_itv
            # print(f"j={j}, pos={pos}: expected_end = {expected_end}, real end = {t[e]}")

            # not valid
            if j == -1 or abs(expected_end - t[e]) > thrd: #t[e] * 0.001:
                print(f"A point is missing at ({k//width-cnt}, {pos})")
                print(f"[INFO] expected_end = {round(expected_end, 2)}, real_end = {t[e]}")
                # print(pointsInfo[j-pos])
                prev_end = expected_end
                valid_ts[pos] = (expected_end-points_itv, expected_end) #(expected_end-touch_itv, expected_end)
                missing += 1

            # valid point, move forward
            else:
                prev_end = t[e]
                j -= 1
                pointsInfo_idx[pos] = s, e
                valid_ts[pos] = (t[s], t[e]) 

            pos -= 1
        
        cnt += 1

        # print(pointsInfo_idx)
        # print(valid_ts)
        # print(len(pointsInfo_idx))
        # print(len(valid_ts))
        new_rows.append(pointsInfo_idx)
        next_start_ts = valid_ts[0][0]
        points_itv, btw_points_itv = summarizeRow2(valid_ts, width)
        touch_itv = points_itv + btw_points_itv

    return new_rows, missing
    


# checks where a given row doesn't have a point at an expected timestamp
# from the beginning of the line (looking for a valid start point)
def validateColumns(t, pointsInfo, k, saved_normal, prev_end_ts, thrd, missing, height):
    points_itv, btw_points_itv, moving_itv = saved_normal
    touch_itv = points_itv + btw_points_itv
    new_rows = []

    j = k
    cnt = 0

    while cnt < height - (k//width):

        pointsInfo_idx = [(-1, -1) for _ in range(width)]
        valid_ts = [(-1, -1) for _ in range(width)]

        expected_start = prev_end_ts + moving_itv
        pos = 0
        s, e = pointsInfo[j]

        # print(f"{cnt} iteration: j = {j}, pos = {pos}")

        # dealing with the beginning of each line(row)
        if abs(expected_start - t[s]) > thrd: #t[s] * 0.00005:
            print(f"A point is missing at ({(j+missing)//width}, {pos})")
            print(f"[INFO] expected_start = {round(expected_start, 2)}, real_start = {t[s]}")
            prev_start = expected_start
            valid_ts[pos] = (expected_start, expected_start+points_itv)
            cnt += 1
        else:
            prev_start = t[s]
            j += 1
            pointsInfo_idx[pos] = s, e
            valid_ts[pos] = (t[s], t[e])
        pos += 1

        # validating the rest of each line(row) with intervals between two points
        while pos < width and j < len(pointsInfo):

            s, e = pointsInfo[j]
                
            expected_start = prev_start + touch_itv
            # print(f"j={j}, pos={pos}: expected_start = {expected_start}, real start = {t[s]}")

            # not valid
            if abs(expected_start - t[s]) > thrd:# t[s] * 0.00005:
                print(f"A point is missing at ({(j+missing)//width}, {pos})")
                print(f"[INFO] expected_start = {round(expected_start, 2)}, real_start = {t[s]}")
                prev_start = expected_start
                valid_ts[pos] = (expected_start, expected_start + points_itv)
                cnt += 1

            # valid point, move forward
            else:
                prev_start = t[s]
                j += 1
                pointsInfo_idx[pos] = s, e
                valid_ts[pos] = (t[s], t[e])

            pos += 1

        cnt += 1
        # print(pointsInfo_idx)
        # print(valid_ts)
        # print(len(pointsInfo_idx))
        # print(len(valid_ts))
        new_rows.append(pointsInfo_idx)
        prev_end_ts = valid_ts[-1][1]
        points_itv, btw_points_itv = summarizeRow2(valid_ts, width)
        touch_itv = points_itv + btw_points_itv

    return new_rows



def updatePointsInfo(t, pointsInfo, width, height, crack):
    print(f"\nFinding rows where points are missing...")
    t_idx, b_idx, middle_rows = validateRows(t, pointsInfo, width, height)
    if crack:
        if t_idx == -1:
            print(f"\n--> No suspicious row detected at the beginning.")
        if b_idx == -1:
            print(f"\n--> No suspicious row detected at the bottom.")

        return t_idx, b_idx, None
    
    missing = 0
    if t_idx == -1:
        print(f"\n--> No suspicious row detected at the beginning.")
    else:
        print(f"\n--> Detected suspicious rows at the beginning, starting from {t_idx//width}th line to the top.")

        # we can tell n1 and n2 are valid rows
        n1, n2 = pointsInfo[t_idx+1:t_idx+1+width].tolist(), pointsInfo[t_idx+1+width:t_idx+1+2*width].tolist()

        # how long it takes from n2 to n1
        moving = t[n2[0][0]] - t[n1[-1][1]]
        # print(f"moving = {moving}")
        next_start_ts = t[n1[0][0]]

        points, btw_points = summarizeRow(t, n1, width)
        saved_normal = (points, btw_points, moving)

        top_rows, missing = validateColumns_rvs(t, pointsInfo, t_idx, saved_normal, next_start_ts, thrd_top)
        middle_rows.extendleft(top_rows)
        # print(top_rows)
        # print(f"len(top_rows)={len(top_rows)}")

    # print(middle_rows)
    # print(f"len(middle_rows)={len(middle_rows)}")

    if b_idx == -1:
        print(f"\n--> No suspicious row detected at the bottom.")
    else:
        print(f"\n--> Detected suspicious rows at the bottom, starting from {(b_idx+missing)//width}th line to the end.")

        # we can tell n1 and n2 are valid rows
        n1, n2 = pointsInfo[b_idx-width:b_idx].tolist(), pointsInfo[b_idx-width*2:b_idx-width].tolist()
    
        # how long it takes from n2 to n1
        moving = t[n1[0][0]] - t[n2[-1][1]]
        # print(f"moving = {moving}")
        prev_end_ts = t[n1[-1][1]]

        points, btw_points = summarizeRow(t, n1, width)
        saved_normal = (points, btw_points, moving)

        bottom_rows = validateColumns(t, pointsInfo, b_idx, saved_normal, prev_end_ts, thrd_bottom, missing, height)
        middle_rows.extend(bottom_rows)


    # print(f"len(middle_rows)={len(middle_rows)}")

    return t_idx, b_idx, np.array(middle_rows)



# crop width*height size of pointsInfo (missing points are shown as (-1, -1))
# to get the data at center only
def cropPoints(width, height, updated, n, top, bottom, left, right):

    h, w, _ = updated.shape

    if h != height:
        raise Exception("The number of rows and your input height doesn't match.")
    if w != width:
        raise Exception("The number of columns and your input width doesn't match.")

    print(f"Original Shape: {updated.shape}")
    # print(updated[-3:, :, :])
    
    e1 = h if bottom == 0 else -bottom
    e2 = w if right == 0 else -right

    pointsInfo = updated[top:e1, left:e2, :]

    nh, nw, _ = pointsInfo.shape
    print(f"New Shape: {pointsInfo.shape}")
    # print(pointsInfo[-3:, :, :])

    pointsInfo = pointsInfo.reshape(nh*nw, 2)
    print(f"Updated Shape: {pointsInfo.shape}")
    
    return pointsInfo



# calculate center of mass of each row
def comCoordinate(row, rx=True):
    if rx:
        start, end = 0, 36
    else:
        start, end = 37, 52
 
    # find local max
    maxI, maxV = start, row[start]
    for i in range(start+1, end+1):
        if row[i] > maxV:
            maxI, maxV = i, row[i]

    # calculate COM
    com = 100*(maxI-start)*maxV
    deno = maxV
    l, r = maxI-1, maxI+1
    while (l >= start and row[l] >= 5) or (r <= end and row[r] >= 5):

        if l >= start and row[l] >= 5:
            com += (l-start)*100*row[l]
            deno += row[l]
            l -= 1
        if r <= end and row[r] >= 5:
            com += (r-start)*100*row[r]
            deno += row[r]
            r += 1
    return round(com/deno, 3)



# find coordinates of each contact&release
# save coordinates info to filename_coor.txt
def findCoordinates(d, pointsInfo, output):

    rx_arr, tx_arr = [], []

    with open(output, 'w+') as f:

        # find the average of COM for each contact&release
        # save information in a text file (tx, rx)
        # rows: rx 37 (0-36), tx 16 (37-52)
        print(f"The length of pointsInfo = {len(pointsInfo)}")
        for i in range(len(pointsInfo)):
            start, end = pointsInfo[i]

            rows = d[start:end, 1:54]
            p_rx, p_tx, n = 0, 0, end-start

            for row in rows:
                p_rx += comCoordinate(row)
                p_tx += comCoordinate(row, False)
            
            rx, tx = round(p_rx/n), round(p_tx/n)
            rx_arr.append(rx)
            tx_arr.append(tx)

            tx_c, rx_c = round(tx/100), round(rx/100)
            f.write(f"{tx_c} {rx_c} {pointsInfo[i]}\n")

    print(f"The length of rx_arr = {len(rx_arr)}")
    print(f"The length of tx_arr = {len(tx_arr)}")

    return rx_arr, tx_arr




if __name__ == '__main__':

    # arguement parser

    parser = argparse.ArgumentParser(description='Program to find contacts(releases)')
    parser.add_argument('filename', help='filename of the data to preprocess')
    parser.add_argument('contacts', type=int, help='the number of total contacts(releases) in a given file')

    parser.add_argument('--notpoint', '-np', action='store_false', help='False if data type is not a point but is a line, pattern, etc.')

    # params for points (required if it's different than default, especially when you want to crop the data)
    parser.add_argument('--width', type=int, default=69, help='the number of total contacts(releases) in a given row')
    parser.add_argument('--height', type=int, default=157, help='the number of total contacts(releases) in a given column')

    parser.add_argument('--thrd_top', '-tt',  type=int, default=10, help='the difference to allow between expected point and real point at top')
    parser.add_argument('--thrd_bottom', '-tb',  type=int, default=30, help='the difference to allow between expected point and real point at bottom')

    parser.add_argument('--left', '-l', type=int, default=0, help='the number of lines not to consider from the left edge')
    parser.add_argument('--right', '-r', type=int, default=0, help='the number of lines not to consider from the right edge')
    parser.add_argument('--top', '-t', type=int, default=0, help='the number of lines not to consider from the top')
    parser.add_argument('--bottom', '-b', type=int, default=0, help='the number of lines not to consider from the bottom')
    

    args = parser.parse_args()
    filename = args.filename
    n, isPoint = args.contacts, args.notpoint
    width, height = args.width, args.height
    thrd_top, thrd_bottom = args.thrd_top, args.thrd_bottom
    left, right, top, bottom = args.left, args.right, args.top, args.bottom


    print(f"\nConverting the input file...")
    start_time = time.time()
    data, max_data = readFile(filename)
    print(f"--> Converted {len(max_data)} rows of data.")
    print(f"--> Elapsed time: {round((time.time() - start_time)/60, 4)}")



    print(f"\nFinding peaks...")
    pointsInfo = pointDetection(max_data, n=n, isPoint=isPoint)
    print(f"--> Detected {len(pointsInfo)} peaks.")
    print(f"--> Elapsed time: {round((time.time() - start_time)/60, 4)}")


    l = os.path.splitext(filename)
    output = f"{l[0]}_out.txt"

    if isPoint and n != len(pointsInfo):
        print(f"\n{n-len(pointsInfo)} points are missing!!!")     

    # validating the found points
    if isPoint:
        print(f"\nValidating the found points...")
        t, cracks = updateTimestamp(np.array(data))
        
        # if the data seems not coherent (has too huge timestampe changes)
        # instead of searching for the exact locations of missing, we recommend manual searching
        if len(cracks) != 0:

            # not updating pointsInfo but checks if it is suspicious
            t_idx, b_idx, _ = updatePointsInfo(t, pointsInfo, width, height, True)
            savePointInfo(pointsInfo, output)
            if t_idx != -1 or b_idx != -1:
                print(f"Manual searching is recommended with info below.")
                print(f"--> Huge timestamp changes are in the indices below.")
                print(cracks)
                if t_idx != -1:
                    print(f"--> Check from index 0 to {t_idx+width-1} index.")
                if b_idx != -1:
                    print(f"--> Check from {b_idx+width-1} index to index {len(pointsInfo)-1}.")

        
        else:
            t_idx, b_idx, updated = updatePointsInfo(t, pointsInfo, width, height, False)

            if top != 0 or bottom != 0 or left != 0 or right != 0:
                print("\nCropping output...")
                pointsInfo = cropPoints(width, height, updated, n, top, bottom, left, right)
                print(f"--> Cropped out the data as you wished.")

            print(f"--> Elapsed time: {round((time.time() - start_time)/60, 4)}")
            pointsInfo = updated.reshape(width*height, 2)
            savePointInfo(pointsInfo, output)


        # updated = updatePointsInfo(t, pointsInfo, width, height)
        print(f"--> Elapsed time: {round((time.time() - start_time)/60, 4)}")
    

    # plt.plot(max_data)
    # plt.plot(pointsInfo, max_data[pointsInfo], "x")
    # plt.show()


    # # Find coordinates and save info
    # print(f"\nFinding coordinates...")
    # output = f"{l[0]}_coor.txt"
    # rx_arr, tx_arr = findCoordinates(d, pointsInfo, output)
    # print(f"--> Calculated {len(pointsInfo)} coordinates in total.")
    # print(f"--> Elapsed time: {round((time.time() - start_time)/60, 4)}")

    # plot the data with its detected (and filtered) peaks
    # plt.plot(max_data)
    # plt.plot(pointsInfo, max_data[pointsInfo], "x")
    # plt.show()

    # # plot the calculated contact&releases
    # plt.scatter(rx_arr, tx_arr, s=1)
    # plt.show()