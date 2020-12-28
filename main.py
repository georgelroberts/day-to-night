"""
Author: G. L. Roberts
Date: 25th December 2020
Description: Run train/test for any techniques
"""

import tensorflow as tf
from absl import flags, app, logging
from models.CycleGAN import CycleGAN
from utils.utils import PlotExamples, load_dataset, get_checkpoint_callback,\
        get_tensorboard_callback

flags.DEFINE_string('technique', 'CycleGAN', '')
flags.DEFINE_boolean('debugging', False, '')
flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_integer('epochs', 200, '')

flags = flags.FLAGS

def main(_):
    train()

def train():
    day_ds, night_ds, combined_ds = load_dataset(flags.batch_size)
    if flags.technique == 'CycleGAN':
        model = CycleGAN()
    else:
        logger.info(f"Technique {flags.technique} not yet implemented")
    model.compile()
    plotter = PlotExamples(day_ds, night_ds, flags.debugging)
    checkpointer = get_checkpoint_callback()
    tb_callback = get_tensorboard_callback()

    if flags.debugging:
        model.fit(combined_ds.take(5), epochs=3,
                callbacks=[plotter, checkpointer, tb_callback])
    else:
        model.fit(combined_ds, epochs=flags.epochs,
                callbacks=[plotter, checkpointer, tb_callback])


if __name__ == "__main__":
    app.run(main)

