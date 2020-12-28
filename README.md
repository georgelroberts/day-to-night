# Day to Night, Night to Day conversion

A tensorflow 2.0 implementation of different unpaired image to image translation
techniques, focussing specifically on conversion between day and night
scenes. This project will firstly use the Synthia dataset, before
moving onto real images.

N.B. All training is run on my local laptop, thus network sizes are small

## Usage

Save two different datasets into data/ (I used SYNTHAI video sequences from http://synthia-dataset.net/downloads/ (CVPR16)
SYNTHIA-SEQS-01-NIGHT and SYNTHIA-SEQS-01-SUMMER). Then run main.py using
python3, setting any flags defined here:

```python
--technique (default CycleGAN)
--debugging (default False)
--batch_size (default 1)
--epochs (default 200)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Distributed under the MIT License. See `LICENSE` for more information.
