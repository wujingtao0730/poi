import os
from ocr import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
import json


def single_pic_proc(image_file):
    image = np.array(Image.open(image_file).convert('RGB'))
    results, contours, img_framed, text_recs = ocr(image)
    return results, contours, img_framed, text_recs


if __name__ == '__main__':
    data = {}
    jsonName = "result.json"
    # image_files = glob('/opt/pytorch/ocr/test_images/*.*')
    image_files = glob('/home/0001/*.*')
    result_dir = './test_result'
    print("\nRecognition Result:\n")
    t = time.time()
    for image_file in sorted(image_files):

        results, contours, img_framed, text_recs = single_pic_proc(image_file)
        point_seq_id = image_file.split('/')[-1].split('.')[0]
        image_id = image_file.split('/')[-1].split('.')[0]
        board_contour = [[1, 2], [3, 4], [5, 6], [7, 8]]
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(img_framed).save(output_file)
        texts = []

        for i in range(len(results)):
            text = {'text': results[i][1], 'contour': contours[i]}
            texts.append(text)
        name = ''
        jsonResult = {'point_seq_id': point_seq_id, 'image_id': image_id, 'board_contour': board_contour, 'texts': texts
                , 'name': name}
        data[point_seq_id] = jsonResult
    poi_json = json.dumps(data, ensure_ascii=False)
    with open(os.path.join('./test_result', jsonName), "w") as f:
        json.dump(poi_json, f, ensure_ascii=False)
    print("Mission complete, it took {:.3f}s".format(time.time() - t))
