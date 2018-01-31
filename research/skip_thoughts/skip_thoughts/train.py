# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train the skip-thoughts model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf

from skip_thoughts import configuration
from skip_thoughts import skip_thoughts_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", None,
                       "File pattern of sharded TFRecord files containing "
                       "tf.Example protos.")
tf.flags.DEFINE_string("train_dir", None,
                       "Directory for saving and loading checkpoints.")

tf.logging.set_verbosity(tf.logging.INFO)


def _setup_learning_rate(config, global_step):
  """Sets up the learning rate with optional exponential decay.

  Args:
    config: Object containing learning rate configuration parameters.
    global_step: Tensor; the global step.

  Returns:
    learning_rate: Tensor; the learning rate with exponential decay.
  """
  if config.learning_rate_decay_factor > 0:
    learning_rate = tf.train.exponential_decay(
        learning_rate=float(config.learning_rate),
        global_step=global_step,
        decay_steps=config.learning_rate_decay_steps,
        decay_rate=config.learning_rate_decay_factor,
        staircase=False)
  else:
    learning_rate = tf.constant(config.learning_rate)
  return learning_rate


def main(unused_argv):
  if not FLAGS.input_file_pattern:
    raise ValueError("--input_file_pattern is required.")
  if not FLAGS.train_dir:
    raise ValueError("--train_dir is required.")

  model_config = configuration.model_config(
      input_file_pattern=FLAGS.input_file_pattern)
  training_config = configuration.training_config()

  tf.logging.info("Building training graph.")
  g = tf.Graph()
  with g.as_default():
    model = skip_thoughts_model.SkipThoughtsModel(model_config, mode="train")
    model.build()

    learning_rate = _setup_learning_rate(training_config, model.global_step)
    optimizer = tf.train.AdamOptimizer(learning_rate)


    # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
    update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))

    # Make sure update_ops are computed before total_loss.
    if update_ops:
      with ops.control_dependencies(update_ops):
        barrier = control_flow_ops.no_op(name='update_barrier')
      total_loss = control_flow_ops.with_dependencies([barrier], total_loss)

    variables_to_train = tf_variables.trainable_variables()

    assert variables_to_train

    gate_gradients=tf_optimizer.Optimizer.GATE_OP
    # Create the gradients. Note that apply_gradients adds the gradient
    # computation to the current graph.
    grads = optimizer.compute_gradients(
        total_loss,
        variables_to_train,
        gate_gradients=gate_gradients,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    grads = tf.contrib.slim.learning.clip_gradient_norms(
        grads,
        training_config.clip_gradient_norm)

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

    with ops.name_scope('train_op'):
      # Make sure total_loss is valid.
      if check_numerics:
        total_loss = array_ops.check_numerics(total_loss,
                                              'LossTensor is inf or nan')

      # Ensure the train_tensor computes grad_updates.
      train_op = control_flow_ops.with_dependencies([grad_updates], total_loss)

    # Add the operation used for training to the 'train_op' collection
    train_ops = ops.get_collection_ref(ops.GraphKeys.TRAIN_OP)
    if train_op not in train_ops:
      train_ops.append(train_op)

    saver = tf.train.Saver()

  tf.contrib.slim.learning.train(
      train_op=train_op,
      logdir=FLAGS.train_dir,
      graph=g,
      global_step=model.global_step,
      number_of_steps=training_config.number_of_steps,
      save_summaries_secs=training_config.save_summaries_secs,
      saver=saver,
      save_interval_secs=training_config.save_model_secs)


if __name__ == "__main__":
  tf.app.run()
