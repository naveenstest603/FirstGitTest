import cv2
import numpy as np
from pytesseract import pytesseract
from PIL import Image
import os
import argparse
import time
import io
import imutils


def pre_process(text):
    # lowercase
    text = str(text).lower()
    # remove tags
    # text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)
    # remove special characters and digits
    # text = re.sub("(\\d|\\W)+", " ", text)
    # text = SpellCheck.correction(text)
    return text


def get_string(img_path_fname, psm_mode, lang, textcolour, hi, wd, result_path_fname, exp_text):
    # pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"
    img_path, fname = os.path.split(img_path_fname)
    lang = str(lang)
    text_colour = str(textcolour)
    try:
        image = Image.open(img_path_fname)
        img = cv2.imread(img_path_fname)
        width_y,length_x,_ = img.shape
        width = int(length_x)
        height = int(width_y)
        print("Width ------------ > ", width)
        print("Height  ------------ > ", height)
        if width < 130 and height > 100:
            _filter = 51
            fx = 2.5
            fy = 2.5
            print(width, " is not sufficient so enhancing with fx=2.5,fy=2.5.")

        elif width > 130 and lang == "eng":
            _filter = 31
            fx = 1.5
            fy = 1.5
            print(width, " is not sufficient so enhancing with fx=1.5,fy=1.5.")

        elif width > 130 and lang == "jpn":
            _filter = 51
            fx = 3.5
            fy = 3.5
            print(width, " is not sufficient so enhancing with fx=3.5,fy=3.5.")

        elif ((width - height) <= 30 and width > height) or (width < 130 and (height < 100)):
            _filter = 71
            fx = 5
            fy = 2.5
            print(width, " is not sufficient so enhancing with fx=5,fy=2.5.")

        img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
        # Convert to gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply dilation and erosion to remove some noise
        kernel = np.ones((1, 1), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        img = cv2.erode(img, kernel, iterations=1)
        if "black" in text_colour:
            img = cv2.threshold(cv2.GaussianBlur(img, (5, 5), 0), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            img = cv2.threshold(cv2.bilateralFilter(img, 5, 75, 75), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            img = cv2.threshold(cv2.medianBlur(img, 3), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        img = cv2.adaptiveThreshold(cv2.GaussianBlur(img, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, _filter, 2)
        img = cv2.adaptiveThreshold(cv2.bilateralFilter(img, 9, 75, 75), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, _filter, 2)
        img = cv2.adaptiveThreshold(cv2.medianBlur(img, 3), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, _filter,
                                    2)
    except Exception as e1:
        print(str(e1))

    processed_image_path_fname = os.path.join(img_path, 'processedImage.png')
    # Write the image after apply opencv to do some ...
    cv2.imwrite(processed_image_path_fname, img)

    img = cv2.imread(processed_image_path_fname)
    h,w,_ = img.shape
    hi = int(hi)
    wd = int(wd)
    y1 = hi
    y2 = h - hi
    x1 = wd
    x2 = w - wd
    image = img[y1: y2, x1: x2]
    cv2.imwrite(processed_image_path_fname, image)

    if lang == "eng":
        os.system("magick mogrify -set density 800 -units pixelsperinch " + processed_image_path_fname)

    ps_mode = [12, 6, 7, 3, 10]
    global final_result
    final_result = []

    try:
        if os.path.isfile(result_path_fname) and result_path_fname is not "null":
            with io.open(result_path_fname, encoding='utf-8', mode='w'):
                print('Result file is erased')
                pass
    except Exception as wrmodeexc:
        print("error : unable to implement write mode : "+str(wrmodeexc))

    if len(psm_mode) > 0:
        try:
            config = ("-l " + lang + " --oem 1 --psm " + str(psm_mode))
            result = pytesseract.image_to_string(Image.open(processed_image_path_fname), config=config)
            print('---------------')
            print('result -> ', result.strip())
            try:
                write_to_file(result,result_path_fname)
            except Exception as wrexc:
                print("error : unable to write unicoded chars : " + str(wrexc))
        except (Exception, IndexError, ValueError) as expn_sing:
            print("Exception Occurred : " + str(expn_sing))
    else:
        try:
            for i in range(len(ps_mode)):
                try:
                    config = ("-l " + lang + " --oem 1 --psm " + str(ps_mode[i]))
                    result = pytesseract.image_to_string(Image.open(processed_image_path_fname), config=config)
                    print('---------------')
                    res = result.lower()
                    if _exp_text is not "null":
                        print('exp_text ', exp_text.encode('utf-8').decode('utf-8'))
                    print('result -> ', result.strip())
                    if exp_text is not "null":
                        print('exp_text is not null')
                        if exp_text.lower() in res:
                            print("Successfully got the text in " + str(ps_mode[i]) + " mode" + " as: " + res)
                            try:
                                write_to_file(res,result_path_fname)
                            except Exception as wrexc:
                                print("error : unable to write unicoded chars : " + str(wrexc))
                            break
                    try:
                        write_to_file(res,result_path_fname)
                    except Exception as wrexc:
                        print("error : unable to write unicoded chars : " + str(wrexc))

                except (Exception, IndexError, ValueError) as expn:
                    print("Exception Occurred pytesseract.image_to_string : " + str(expn))
        except (Exception, IndexError, ValueError) as expn:
            print("Exception Occurred : " + str(expn))
    try:
        with io.open(result_path_fname, encoding='utf-8', mode='r') as the_file:
            print("Extracted Text : ")
            print(the_file.read())
    except Exception:
        print("error : unable to read extracted text ")


def find_rect(in_image, cropped_path, _dimension_x, _dimension_y):
    img_path, fname = os.path.split(cropped_path)
    img = cv2.imread(in_image, cv2.IMREAD_COLOR)
    copy_image = img.copy()
    copy_input_image = os.path.join(img_path, 'copy_in_image.png')
    cv2.imwrite(copy_input_image, copy_image)
    time.sleep(0.500)
    os.system("magick mogrify -set density 1500x1024 -units pixelsperinch " + "copy_in_image.png")
    img = cv2.imread("copy_in_image.png", cv2.IMREAD_COLOR)
    img = cv2.resize(img, (int(_dimension_x), int(_dimension_y)))           #INTER_AREA, INTER_NEAREST
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(gray, 30, 200)
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screen_Cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screen_Cnt = approx
            break
    if screen_Cnt is None:
        print("No contour detected")
    mask = np.zeros(gray.shape, np.uint8)
    try:
        cv2.drawContours(mask, [screen_Cnt], 0, 255, -1, )
        new_image = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(cropped_path, new_image)
        print('Found rect successfully..')
    except (Exception, IndexError, ValueError) as exeptn:
        print("Error : Unable to print or write unicode chars : " + str(exeptn))
    return cropped_path


def write_to_file(obj, result_path_fname):
    try:
        if os.path.isfile(result_path_fname):
            with io.open(result_path_fname, encoding='utf-8', mode='a+') as the_file:
                the_file.write(obj.strip() + '\n')
        else:
            print("Result file not found ..")
    except (Exception, IndexError, ValueError) as exptn:
        print("Error : Unable to print or write unicode chars : " + str(exptn))
    return result_path_fname


def transform(pos):
    # This function is used to find the corners of the object and the dimensions of the object
    pts = []
    n = len(pos)
    for i in range(n):
        pts.append(list(pos[i][0]))

    sums = {}
    diffs = {}
    tl = tr = bl = br = 0
    for i in pts:
        x = i[0]
        y = i[1]
        sum = x + y
        diff = y - x
        sums[sum] = i
        diffs[diff] = i
    sums = sorted(sums.items())
    diffs = sorted(diffs.items())
    n = len(sums)
    try:
        rect = [sums[0][1], diffs[0][1], diffs[n - 1][1], sums[n - 1][1]]
    # top-left   top-right   bottom-left   bottom-right
        h1 = np.sqrt((rect[0][0] - rect[2][0]) ** 2 + (rect[0][1] - rect[2][1]) ** 2)  # height of left side
        h2 = np.sqrt((rect[1][0] - rect[3][0]) ** 2 + (rect[1][1] - rect[3][1]) ** 2)  # height of right side
        h = max(h1, h2)

        w1 = np.sqrt((rect[0][0] - rect[1][0]) ** 2 + (rect[0][1] - rect[1][1]) ** 2)  # width of upper side
        w2 = np.sqrt((rect[2][0] - rect[3][0]) ** 2 + (rect[2][1] - rect[3][1]) ** 2)  # width of lower side
        w = max(w1, w2)
    except (IndexError, ValueError):
        print("IndexError: list index out of range.--No Rectangle box")
        w= 'null'
        h='null'
        rect = 'Null'

    return int(w), int(h), rect


def crop_image(in_image, cropped_path):
    im = Image.open(in_image)
    length_x, width_y = im.size
    width = int(str(length_x))
    height = int(str(width_y))
    global fx
    global fy
    print(" Screen shot Width ------------ > ", width)
    print(" Screen shot Height  ---------- > ", height)
    if width < 800:
        fx = 1.5
        fy = 1.5
    else:
        fx = 0.7
        fy = 0.7
    dir_path = os.path.realpath(in_image)
    img = cv2.imread(dir_path)
    img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(gray, 100, 200, apertureSize=5)
    _, thresh = cv2.threshold(edge, 1, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    pos = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > max_area:
            max_area = area
            pos = i
    peri = cv2.arcLength(pos, True)
    approx = cv2.approxPolyDP(pos, 0.02 * peri, True)
    w, h, arr = transform(approx)

    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    pts1 = np.float32(arr)
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(img, M, (w, h))
    img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    try:
        image = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    # cv2.imshow('OUTPUT',image)
        cv2.imwrite(cropped_path, image)
        print('Cropped successfully..')
    except Exception as ee:
        print("Exception :" + str(ee))
    time.sleep(0.300)
    return cropped_path


def crop_image_max_rect(in_image,cropped_path):
    dir_path = os.path.realpath(in_image)
    print(dir_path)
    img = cv2.imread(in_image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # convert to grayscale
    # threshold to get just the signature (INVERTED)
    retval, thresh_gray = cv2.threshold(gray, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # Find object with the biggest bounding box
    mx = (0, 0, 0, 0)  # biggest bounding box so far
    mx_area = 0
    for cont in contours:
        x, y, w, h = cv2.boundingRect(cont)
        area = w * h
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
    x, y, w, h = mx
    # Output to files
    roi = img[y+5:y + h-5, x+6:x + w-6]
    cv2.imwrite(cropped_path, roi)
    time.sleep(0.300)
    print('Cropped successfully..')
    return cropped_path


def crop_custom(top, down, left, right, custom_path_fname,custom_crop):
    custom_path, _ = os.path.split(custom_path_fname)
    img = cv2.imread(_path)
    h,w,_ = img.shape
    y1 = int(h * (top/100))
    y2 = h - int(h * (down/100))
    x1 = int(w * (left/100))
    x2 = w - int(w * (right/100))
    image = img[y1:y2, x1:x2]
    custom_cropped_path_fname = os.path.join(custom_path,custom_crop+"custom_cropped.png")
    cv2.imwrite(custom_cropped_path_fname,image)
    return custom_cropped_path_fname


def parse_args():
    # this method is used for parse command line argument to the program.
    parser = argparse.ArgumentParser(description='IMAGE FILE FOR THE IMAGE PROCESSING')
    parser.add_argument('--input_image', dest='input_image', help='Input Image File Path',
                        default='', required=True)
    parser.add_argument('--cropped_image', dest='cropped_image', help='Cropped Image File Path',
                        default='', required=True)
    parser.add_argument('--action', dest='action', help='rect_text, get_String to be Specified', default='', required=True)
    parser.add_argument('--result', dest='result', help='Result.txt file path to be Specified for the extracted string',
                        default='null',
                        required=True)
    parser.add_argument('--mode', dest='mode', help='psm mode to be Specified for char sequence 12,6,7,3,10', default='',
                        required=False)
    parser.add_argument('--lang', dest='lang', help='language to be Specified eng', default='eng',
                        required=False)
    parser.add_argument('--color', dest='color', help='Color Black or Grey to be Specified ', default='black',
                        required=False)
    parser.add_argument('--height', dest='height', help='Height-int to be Specified to crop the sides of processed image', default='0',
                        required=False)
    parser.add_argument('--width', dest='width', help='Width-int to be Specified to crop the sides of processed image', default='0',
                        required=False)
    parser.add_argument('--rect_size', dest='rect_size', help='rect_size to be specified for the specific shape as small, big',
                        default='', required=False)
    parser.add_argument('--crop_top', dest='crop_top', help='crop top to be specified to crop input image as 10 or 20 etc', default='0',
                        required=False)
    parser.add_argument('--crop_down', dest='crop_down',
                        help='crop down to be specified to crop input image as 10 or 20 etc',
                        default='0', required=False)
    parser.add_argument('--crop_left', dest='crop_left',
                        help='crop left to be specified to crop input image as 10 or 20 etc',
                        default='0', required=False)
    parser.add_argument('--crop_right', dest='crop_right',
                        help='crop right to be specified to crop input image as 10 or 20 etc',
                        default='0', required=False)
    parser.add_argument('--exp_text', dest='exp_text',
                        help='Expected text to be specified for the text extraction in different modes',
                        default='null', required=False)
    parser.add_argument('--dimension_x', dest='dimension_x', help='dimension_x to be specified to resize input image '
                                                            'value should be int ex: 600', default='600',
                        required=False)
    parser.add_argument('--dimension_y', dest='dimension_y',
                        help='dimension Y to be specified to resize the image as the value should be in ex: 400',
                        default='400', required=False)
    argss = parser.parse_args()
    return argss


if __name__ == '__main__':
    start_time = time.time()
    is_input_file_present = False
    args = parse_args()
    _mode = args.mode
    _lang = args.lang
    _color = args.color
    _height = args.height
    _width = args.width
    _action = args.action
    _result = args.result
    _path = args.input_image
    _cropped_path = args.cropped_image
    _exp_text = args.exp_text
    __rect_size = args.rect_size
    _crop_top = args.crop_top
    _crop_down = args.crop_down
    _crop_left = args.crop_left
    _crop_right = args.crop_right
    _dimension_x = args.dimension_x
    _dimension_y = args.dimension_y

    for x in range(4):
        if os.path.isfile(_path):
            print("Got the --input_image file in specified location ")  # with matrix :" + str(cv2.imread(_path).size))
            is_input_file_present = True
            break
        else:
            time.sleep(1)
            print("Looking for the --input_image file In Loop ")  # : " + str(cv2.imread(_path) is None))

    if is_input_file_present is True:
        c_flag = True
        c_top = int(_crop_top)
        c_down = int(_crop_down)
        c_left = int(_crop_left)
        c_right = int(_crop_right)
        crops = []

        if c_top or c_down or c_left or c_right:
            print(" ---------single custom crop -----------" + str(c_top) + "," + str(c_down))
            _path = crop_custom(c_top, c_down, c_left, c_right,_path ,"single_variant_")

        if str(_action).lower() == "rect_text":

            if len(__rect_size) != 0:

                if "small" == __rect_size:
                    print("selected rect box size is small sized rectangle")
                    cropped_img_path = find_rect(_path, _cropped_path, _dimension_x, _dimension_y)
                    cropped_img_path = crop_image(cropped_img_path, _cropped_path)
                    try:
                        get_string(cropped_img_path, _mode, _lang, _color, _height, _width, _result, _exp_text)
                    except (Exception, IndexError, ValueError) as e:
                        print("Error : Expected String not found " + str(e))

                elif "big" == __rect_size:
                    print("selected rect box is big sized rectangle")
                    cropped_img_path = crop_image_max_rect(_path,_cropped_path)
                    try:
                        get_string(cropped_img_path, _mode, _lang, _color, _height, _width, _result, _exp_text)
                    except (Exception, IndexError, ValueError) as e:
                        print("Error : Expected String not found" + str(e))

                else:
                    print("selected rect size is not correct ")
                    exit(1)
            else:
                print("valid --rect_size should be assigned as small or big")
                exit(1)

            print('------ Done -------')
            print("--- %s seconds ---" % (time.time() - start_time))

        elif str(_action).lower() == "get_string":
            get_string(_path, _mode, _lang, _color, _height, _width, _result, _exp_text)
            # print("Processed Text : " + str(out.strip().encode('utf-8', 'ignore').strip()))
            print('------ Done -------')
            print("--- %s seconds ---" % (time.time() - start_time))
        else:
            print("action is not defined properly ")

    else:
        print("Image is not ready to read / not available..")


        # python3 ImageProcess_Mac_os_stable_V1.py --action gettext --crop_shape rect --cropped_image cropped.png --result /Users/nnauvusali/Downloads/result --input_image /Users/nnauvusali/Downloads/images16.png