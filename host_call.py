"""
Copied from tensor2tensor library.

tensor2tensor/tensor2tensor/utils/t2t_model.py

"""

import tensorflow as tf
import collections
import six

_already_logged = set()


def _eager_log(level, *args):
  if tf.contrib.eager.in_eager_mode() and args in _already_logged:
    return
  _already_logged.add(args)
  getattr(tf.logging, level)(*args)


def log_info(*args):
  _eager_log("info", *args)


def log_debug(*args):
  _eager_log("debug", *args)


def create_host_call(model_dir):
  """Construct a host_call writing scalar summaries.
  Args:
    model_dir: String containing path to train
  Returns:
    (fn, args) Pair to be called by TPUEstimator as the host_call.
  """
  graph = tf.get_default_graph()
  summaries = graph.get_collection(tf.GraphKeys.SUMMARIES)
  gs_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
  summary_kwargs = collections.OrderedDict()
  for t in summaries:
    # TODO(aidangomez): enable ImageSummary support when we have a faster method
    # see @shibow's comment in cl/202344570
    if t.op.type not in ["ScalarSummary"]:
      tf.logging.warn("Ignoring unsupported tf.Summary type %s" % t.op.type)
      continue

    name = t.op.name
    tensor = t.op.inputs[1]
    if t.op.type == "ScalarSummary":
      assert tensor.shape.is_compatible_with([])
      if tensor.dtype == tf.int64:
        tensor = tf.to_int32(tensor)
      summary_kwargs["ScalarSummary" + name] = tf.reshape(tensor, [1])
    elif t.op.type == "ImageSummary":
      # TODO(aidangomez): as we move to support more types, update
      # common_layers.tpu_safe_image_summary
      if tensor.dtype != tf.float32:
        tf.logging.warn(
            "Currently T2T on TPU only supports ImageSummary of "
            "tf.float32-type Tensors. Skipping Tensor "
            "%s with dtype %s..." % (tensor.name, tensor.dtype))
        continue
      # tensor = tf.to_float(tensor)
      summary_kwargs["ImageSummary" + name] = tensor
  # When no supported summaries are found, don't create host_call. Otherwise,
  # TPU outfeed queue would enqueue global_step while host_call doesn't dequeue
  # it, eventually causing hang.
  if not summary_kwargs:
    return None
  summary_kwargs["global_step"] = gs_t
  log_info("summary_kwargs %s" % str(summary_kwargs))

  def host_call_fn(**kwargs):
    """Training host call. Creates summaries for training metrics.
    Args:
      **kwargs: Dict of {str: Tensor} , with `Tensor` of shape `[batch]`. Must
        contain key "global_step" with value of current global_step Tensor.
    Returns:
      List of summary ops to run on the CPU host.
    """
    gs = tf.to_int64(kwargs.pop("global_step")[0])
    with tf.contrib.summary.create_file_writer(model_dir).as_default():
      with tf.contrib.summary.always_record_summaries():
        # We need to use tf.contrib.summary in order to feed the `step`.
        for name, value in sorted(six.iteritems(kwargs)):
          if name.startswith("ScalarSummary"):
            name = name[len("ScalarSummary"):]
            tf.contrib.summary.scalar(
                name, tf.reduce_mean(tf.to_float(value)), step=gs)
          elif name.startswith("ImageSummary"):
            name = name[len("ImageSummary"):]
            tf.contrib.summary.image(name, value, step=gs)

        return tf.contrib.summary.all_summary_ops()

  return (host_call_fn, summary_kwargs)


def remove_summaries():
  """Remove summaries from the default graph."""
  g = tf.get_default_graph()
  key = tf.GraphKeys.SUMMARIES
  log_debug("Remove summaries %s" % str(g.get_collection(key)))
  del g.get_collection_ref(key)[:]
  assert not g.get_collection(key)