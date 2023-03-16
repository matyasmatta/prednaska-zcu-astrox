import argparse
import time

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import json
import os


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  count = 0
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)],
                   outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10),
              '%s\n%.2f' % (count, obj.score),
              fill='red')
    count += 1


def ai_model(image_path):

    open(r"model\labelmap.txt")
    labels = r'model\labelmap.txt'
    interpreter = make_interpreter(r'model\edgetpu.tflite')
    interpreter.allocate_tensors()

    image = Image.open(image_path)
    _, scale = common.set_resized_input(
        interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
    print(scale)

    # print('----INFERENCE TIME----')
    # print('Note: The first inference is slow because it includes', 'loading the model into Edge TPU memory.')
    for _ in range(2):
        start = time.perf_counter()
        interpreter.invoke()
        inference_time = time.perf_counter() - start
        objs = detect.get_objects(interpreter, 0, scale)
        print('%.2f ms' % (inference_time * 1000))

    # print('-------RESULTS--------')
    if not objs:
        print('No objects detected')
    counter_for_ai_output = 0
    ai_output = {}
    for obj in objs:
        #print(labels.get(obj.id, obj.id))
        print('  id:    ', obj.id)
        print('  score: ', obj.score)
        print('  bbox:  ', obj.bbox)

        # obj.bbox needs to be converted into a dictionary
        bbox = obj.bbox
        score = obj.score
        ai_output[counter_for_ai_output] = {}
        ai_output[counter_for_ai_output]['xmin'] = bbox.xmin
        ai_output[counter_for_ai_output]['ymin'] = bbox.ymin
        ai_output[counter_for_ai_output]['xmax'] = bbox.xmax
        ai_output[counter_for_ai_output]['ymax'] = bbox.ymax
        ai_output[counter_for_ai_output]['accuracy'] = score

        counter_for_ai_output += 1
    image = image.convert('RGB')
    draw_objects(ImageDraw.Draw(image), objs, labels)    
        
    # image.show()
    if os.path.exists('meta.jpg') == True:
        os.remove('meta.jpg')
    image.save('meta.jpg')

    with open('ai_output.json', 'w', encoding='utf-8') as f:
        json.dump(ai_output, f, ensure_ascii=False, indent=4)
    return ai_output


if __name__ == '__main__':
    data = ai_model('zchop.meta.x000.y000.n011.jpg')
    print(data)