> tensorflow里一种转pb及lite的方式

```python
import tensorflow as tf
from mobilenet import *

input = tf.placeholder(tf.float32,shape = (1,250,250,3), name = 'input')
# img = tf.placeholder(name="img", dtype=tf.float32, shape=(1, 64, 64, 3))
# var = tf.get_variable("weights", dtype=tf.float32, shape=(1,64,64,3))
# val = img + var
val = fcn2(input)
pb_file_path = './graph.pb'

def canonical_name(x):
  return x.name.split(":")[0]

out = tf.identity(val, name="out")
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  saver.restore(sess, './record/fcn-final.tfmodel')
  out_tensors = [out]
  frozen_graphdef = tf.graph_util.convert_variables_to_constants(
      sess, sess.graph_def, map(canonical_name, out_tensors))
  # tf.train.write_graph(frozen_graphdef, '.', 'graph.pb', as_text=False)
  with tf.gfile.FastGFile(pb_file_path, mode='wb') as f:
    f.write(frozen_graphdef.SerializeToString())
  # tflite_model = tf.contrib.lite.toco_convert(
      # frozen_graphdef, [input], out_tensors)
  # open("converted_model.tflite", "wb").write(tflite_model)
```
