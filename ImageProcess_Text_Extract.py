import cv2
import numpy as np
import pytesseract
from PIL import Image
import os
import argparse
import time
import io


def pre_process(text):
    # lowercase
    text = str(text).lower()
    return text


def get_string(img_path_fname, mode, lang, crop_icon, cropTopBtm, textcolour, hi, wd,result_path_fname,exp_text):
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

    processed_image_path_fname = os.path.join(img_path,'processedImage.png')
    # Write the image after apply opencv to do some ...
    cv2.imwrite(processed_image_path_fname, img)

    img = cv2.imread(processed_image_path_fname)
    h,w,_ = img.shape
    hi = int(hi)
    wd = int(wd)
    if "true" in cropTopBtm:
        y1 = int(h * 0.50)
        y2 = h - int(h * 0.30)
        x1 = 20
        x2 = w-20
    elif "true" in crop_icon:
        y1 = hi
        y2 = h - hi
        x1 = int(w * 0.13)
        x2 = w - wd
    else:
        y1 = hi
        y2 = h - hi
        x1 = wd
        x2 = w - wd
    image = img[y1: y2, x1: x2]
    cv2.imwrite(processed_image_path_fname, image)

    if lang == "jpn":
        os.system("magick mogrify -set density 1000 -units pixelsperinch " + processed_image_path_fname)
    elif lang == "eng":
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

    if len(_mode) > 0:
        try:
            config = ("-l " + lang + " --oem 1 --psm " + str(_mode))
            #result = pytesseract.image_to_string(Image.open(img_path + '/processedImage.png'), config=config)
            result = pytesseract.image_to_string(Image.open(processed_image_path_fname), config=config)
            print('result ')
            print('---------------')
            #final_result.append(result + '\n')
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
                    print('before calling img to string')
                    #result = pytesseract.image_to_string(Image.open(img_path + '/processedImage.png'), config=config)
                    result = pytesseract.image_to_string(Image.open(processed_image_path_fname), config=config)
                    print('result ')
                    print('---------------')
                    #result.encode('ascii', 'ignore').decode('ascii')
                    #res = result.lower()
                    res = result.lower()
                    #print('exp_text ',exp_text.encode("utf-8"))
                    print('exp_text ',exp_text.encode('utf-8').decode('utf-8'))

                    print('result ',result)
                    if exp_text is not "null":
                        print('exp_text is not null')
                        if exp_text.lower() in res:
                            if lang =='jpn':
                                print('jpn language')
                            else:
                                print("Successfully got the text in " + str(ps_mode[i]) + " mode" + " as: " + res)
                            #final_result.append(result + '\n')
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
    #config = '--user-words words.txt config.txt'
    try:
        with io.open(result_path_fname, encoding='utf-8', mode='r') as the_file:
            #print("Extracted Text : " + str(obj))
            print("Extracted Text : ")
            print(the_file.read())
    except Exception:
        print("error : unable to read extracted text ")

    try:
        if int(_crop_variant) > 0:
            with open(result_path_fname, encoding='utf-8', mode='r') as f:
                final_result = f.readlines()
                # print("print output : " + str(final_result))
                pre_process(str(final_result))
    except Exception:
        print("error : Unable to read lines for the return value")


def write_to_file(obj,result_path_fname):
    try:
        if os.path.isfile(result_path_fname):
            with io.open(result_path_fname, encoding='utf-8', mode='a+') as the_file:
                #print("Extracted Text : " + str(obj))
                the_file.write(obj + '\n')
        else:
            print("Result file not found ..")
    except (Exception, IndexError, ValueError) as exptn:
        print("Error : Unable to print or write unicode chars : " + str(exptn))
    #return str(obj)
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


def crop_image(in_image,cropped_path):
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
    # print(dir_path)

    img = cv2.imread(dir_path)
    img = cv2.resize(img, None, fx=fx, fy=fy, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edge = cv2.Canny(gray, 100, 200, apertureSize=5)

    _, thresh = cv2.threshold(edge, 1, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #cv2.drawContours(img, contours, -1, [0, 255, 0], 2)
    # cv2.imshow('Contours',img)
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

    image, contours, hierarchy = cv2.findContours(thresh_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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


def get_toggle_status(cropped_path):
    val = False
    head, tail = os.path.split(cropped_path)
    img = cv2.imread(cropped_path)
    w = img.shape[1]
    h = img.shape[0]
    image = img[10: h - 5, int(w * 0.75):w - 10].copy()
    image = Image.fromarray(image)
    image.save(head + '/Toggle.png')
    RGBim = Image.open(head + '/Toggle.png').convert('RGB')

    HSVim = RGBim.convert('HSV')

    # Make numpy versions
    RGBna = np.array(RGBim)
    HSVna = np.array(HSVim)

    # Extract Hue
    H = HSVna[:, :, 0]

    # Find all green pixels, i.e. where 100 < Hue < 140
    lo, hi = 100, 140
    # Rescale to 0-255, rather than 0-360 because we are using uint8
    lo = int((lo * 255) / 360)
    hi = int((hi * 255) / 360)
    green = np.where((H > lo) & (H < hi))

    # Make all green pixels black in original image
    RGBna[green] = [0, 0, 0]

    count = green[0].size
    print("Pixels matched: {}".format(count))
    if count > 1000:
        val = True
        print("found green toggle button in ON state ", val)
    else:
        print("found toggle button in OFF state ", val)

    return val



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
    parser.add_argument('--action', dest='action', help='Action Name getText , getTextJpn, getTextVertRect, '
                                                        'getToggleStatus, crop_rect, get_String to be '
                                                        'Specified', default='', required=True)
    parser.add_argument('--result', dest='result', help='Result file path to be Specified for the extracted string',
                        default='null',
                        required=True)
    parser.add_argument('--mode', dest='mode', help='Mode to be Specified for char sequence 12,6,7,3,10', default='',
                        required=False)
    parser.add_argument('--lang', dest='lang', help='language to be Specified eng or jpn ', default='eng',
                        required=False)
    parser.add_argument('--ver_crop', dest='ver_crop', help='Verticle Crop to be Specified True or False',
                        default='false', required=False)
    parser.add_argument('--color', dest='color', help='Color Black or Grey to be Specified ', default='black',
                        required=False)
    parser.add_argument('--height', dest='height', help='Height-int to be Specified to crop the sides', default='0',
                        required=False)
    parser.add_argument('--width', dest='width', help='Width-int to be Specified to crop the sides', default='0',
                        required=False)
    parser.add_argument('--crop_icon', dest='crop_icon', help='crop_icon to be Specified to crop the icon as True or '
                                                              'False ', default='false', required=False)
    parser.add_argument('--crop_shape', dest='crop_shape', help='crop shape to be specified for the specific shape',
                        default='', required=False)
    parser.add_argument('--crop_top', dest='crop_top', help='crop top to be specified to crop input image as the '
                                                            'value should be in percentage', default='0',
                        required=False)
    parser.add_argument('--crop_down', dest='crop_down',
                        help='crop down to be specified to crop input image as the value should be in percentage',
                        default='0', required=False)
    parser.add_argument('--crop_left', dest='crop_left',
                        help='crop left to be specified to crop input image as the value should be in percentage',
                        default='0', required=False)
    parser.add_argument('--crop_right', dest='crop_right',
                        help='crop right to be specified to crop input image as the value should be in percentage',
                        default='0', required=False)
    parser.add_argument('--exp_text', dest='exp_text',
                        help='Expected text to be specified for the text extraction in different modes',
                        default='null', required=False)
    parser.add_argument('--crop_variant', dest='crop_variant',
                        help='crop_variant to be specified for the 3 different custom crops by the variant value',
                        default='0', required=False) # amount by which crop will be effected  10, 20 etc

    argss = parser.parse_args()

    return argss


if __name__ == '__main__':

    start_time = time.time()
    # userPath = Path.home()
    is_input_file_present = False
    args = parse_args()
    _mode = args.mode
    _lang = args.lang
    _ver_crop = args.ver_crop
    _color = args.color
    _height = args.height
    _width = args.width
    _action = args.action
    _result = args.result
    _crop_icon = args.crop_icon
    _path = args.input_image
    _cropped_path = args.cropped_image
    _exp_text = args.exp_text
    global _crop_variant
    _crop_variant = args.crop_variant
    _crop_shape = args.crop_shape
    _crop_top = args.crop_top
    _crop_down = args.crop_down
    _crop_left = args.crop_left
    _crop_right = args.crop_right



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
        auto_crop_up = False
        auto_crop_down = False
        auto_crop_middle = False
        c_top = int(_crop_top)
        c_down = int(_crop_down)
        c_left = int(_crop_left)
        c_right = int(_crop_right)
        crops = []
        if int(_crop_variant) > 0:
            for c_value in range(3):
                if c_value == 0:
                    _croptop = int(_crop_top) - int(_crop_variant)
                    _cropdown = int(_crop_down) + int(_crop_variant)
                    print("in up variant crop" + str(_croptop) + "" + str(_cropdown))
                    variant_name = "up_variant_"
                if c_value == 1:
                    _croptop = int(_crop_top)
                    _cropdown = int(_crop_down)
                    print("in middle variant crop" + str(_croptop) + "" + str(_cropdown))
                    variant_name = "middle_variant_"
                if c_value == 2:
                    _croptop = int(_crop_top) + int(_crop_variant)
                    _cropdown = int(_crop_down) - int(_crop_variant)
                    print("in down variant crop" + str(_croptop) + "" + str(_cropdown))
                    variant_name = "down_variant_"
                _crop_variant_path = crop_custom(_croptop, _cropdown, c_left, c_right, _path, variant_name)
                crops.append(_crop_variant_path)
                print(str(crops))
        else:
            print(" ---------single custom crop -----------" + str(c_top) + "," + str(c_down))
            _path = crop_custom(c_top, c_down, c_left, c_right,_path ,"single_variant_")

        if str(_action).lower() == "gettext":

            if len(_crop_shape) == 0:
                if int(_crop_variant) > 0:
                    for index in range(len(crops)):
                        print("cropped path variant : " + str(crops[index]))
                        cropped_img_path = crop_image(str(crops[index]),_cropped_path)
                        try:
                            get_string(cropped_img_path, _mode, _lang, _crop_icon, _ver_crop, _color, _height, _width,_result,_exp_text)
                            #print("return value here : "+out)
                        except (Exception, IndexError, ValueError) as e:
                            print("Error : Expected String not found in "+str(index)+" Iteration" + str(e))
                            continue
                        if _exp_text.lower() in str(final_result).lower():
                            break
                    for item in range(len(crops)):
                        os.remove(str(crops[item]))
                else:
                    cropped_img_path = crop_image(_path,_cropped_path)
                    try:
                        get_string(cropped_img_path, _mode, _lang, _crop_icon, _ver_crop, _color, _height, _width,_result,_exp_text)
                        #print("return value here : " + out)
                    except (Exception, IndexError, ValueError) as e:
                        print("Error : Expected String not found " + str(e))

            else:
                if "rect" == _crop_shape:
                    print("selected crop shape is rectangle")
                    if int(_crop_variant) > 0:
                        for index in range(len(crops)):
                            print("cropped path variant : " + str(crops[index]))
                            cropped_img_path = crop_image_max_rect(str(crops[index]),_cropped_path)
                            try:
                                get_string(cropped_img_path, _mode, _lang, _crop_icon, _ver_crop, _color, _height,
                                                 _width,_result,_exp_text)
                            except (Exception, IndexError, ValueError) as e:
                                print("Error : Expected String not found in " + str(index) + " Iteration" + str(e))
                                continue
                            if _exp_text.lower() in str(final_result).lower():
                                break
                        for item in range(len(crops)):
                            os.remove(str(crops[item]))
                    else:
                        cropped_img_path = crop_image_max_rect(_path,_cropped_path)
                        try:
                            get_string(cropped_img_path, _mode, _lang, _crop_icon, _ver_crop, _color, _height,
                                             _width,_result,_exp_text)
                        except (Exception, IndexError, ValueError) as e:
                            print("Error : Expected String not found" + str(e))
                else:
                    print("selected crop shape is not correct ")
                    exit(1)

            # out = out.encode(encoding='utf-8', errors='ignore')
            # out = out.encode('ascii', 'ignore').decode('ascii', 'ignore')
            # print("Processed Text : " + str(out))
            print('------ Done -------')
            print("--- %s seconds ---" % (time.time() - start_time))
        elif str(_action).lower() == "crop_rect":
            cropped_img_path = crop_image_max_rect(_path,_cropped_path)

        elif str(_action).lower() == "get_string":
            get_string(cropped_img_path, _mode, _lang, _crop_icon, _ver_crop, _color, _height, _width,_result,_exp_text)
            # print("Processed Text : " + str(out.strip().encode('utf-8', 'ignore').strip()))
            print('------ Done -------')
            print("--- %s seconds ---" % (time.time() - start_time))
        elif str(_action).lower() == "gettextvertrect":
            cropped_img_path = crop_image_max_rect(_path,_cropped_path)
            get_string(cropped_img_path, 10, "eng", "false" "true", "black", 0, 0,_result,_exp_text)
            # print("Processed Text : " + str(out.strip().encode('utf-8', 'ignore').strip()))
            print('------ Done -------')
            print("--- %s seconds ---" % (time.time() - start_time))

        elif str(_action).lower() == "gettextjpn":
            cropped_img_path = crop_image_max_rect(_path,_cropped_path)
            get_string(cropped_img_path, _mode, "jpn", _crop_icon, _ver_crop, _color, _height, _width,_result,_exp_text)
            # print("Processed Text : " + str(out))
            print('------ Done -------')
            print("--- %s seconds ---" % (time.time() - start_time))

        elif str(_action).lower() == "gettogglestatus":
            get_toggle_status(_cropped_path)
            print("--- %s seconds ---" % (time.time() - start_time))
    else:
        print("Image is not ready to read / not available..")
