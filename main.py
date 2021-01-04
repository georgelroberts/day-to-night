"""
Author: G. L. Roberts
Date: 25th December 2020
Description: Run train/test for any techniques
"""

import tensorflow as tf
from absl import flags, app, logging
from models.CycleGAN import CycleGAN
from models.MUNIT import MUNIT
from utils.utils import load_dataset, get_checkpoint_callback,\
        get_tensorboard_callback
from utils.CycleGANUtils import PlotExamplesCycleGAN, RecalcCycleWeight
from utils.MUNITUtils import PlotExamplesMUNIT

flags.DEFINE_string('technique', 'MUNIT', '')
flags.DEFINE_boolean('debugging', False, '')
flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_integer('epochs', 200, '')
flags.DEFINE_string('run_name', 'lowerReconLoss', '')

flags = flags.FLAGS

def main(_):
    train()

def train():
    day_ds, night_ds, combined_ds = load_dataset(flags.batch_size)
    callbacks = []
    if flags.technique == 'CycleGAN':
        model = CycleGAN()
        plotter = PlotExamplesCycleGAN(day_ds, night_ds,
                flags.debugging, flags.run_name)
        cycle_weight_callback = RecalcCycleWeight()
        callbacks.extend([plotter, cycle_weight_callback])
    elif flags.technique == 'MUNIT':
        plotter = PlotExamplesMUNIT(day_ds, night_ds,
                flags.debugging, flags.run_name)
        callbacks.extend([plotter])
        model = MUNIT()
    else:
        logger.info(f"Technique {flags.technique} not yet implemented")
    checkpointer = get_checkpoint_callback(flags.run_name)
    tb_callback = get_tensorboard_callback(flags.run_name)
    callbacks.extend([checkpointer, tb_callback])
    model.compile()
    # model.load_weights(f'tmp/{run_name}/{run_name}.ckpt')

    if flags.debugging:
        model.fit(combined_ds.take(5), epochs=3,
                callbacks=callbacks)
    else:
        model.fit(combined_ds.take(100), epochs=flags.epochs,
                callbacks=callbacks)


if __name__ == "__main__":
    app.run(main)

