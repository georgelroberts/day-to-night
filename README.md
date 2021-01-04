# Day to Night, Night to Day conversion

A tensorflow 2.0 implementation of different unpaired image to image translation
techniques, focussing specifically on conversion between day and night
scenes. This project will firstly use the Synthia dataset, before
moving onto real images.

N.B. All training is run on my local laptop, thus network sizes are small.

## Techniques

CycleGAN: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (https://arxiv.org/pdf/1703.10593.pdf).
	This work introduces the concept of 'cycle consistency', whereby we minimize the difference between the real image 
	and the image after being passed through the encoder and then the decoder.
	
MUNIT: Multimodal Unsupervised Image-to-Image Translation (https://arxiv.org/pdf/1804.04732.pdf).
	This is based on a disentangled representation. The encoder consists of two separate
	networks, the style encoder (which should contain the concept of 'day' or 'night') 
	and the content encoder (which should encode the physical features in the image). By mixing these in the decoder,
	we can generate new images. This uses both image reconstruction loss and latent (style/encoder) reconstruction loss,
	alongside the adversarial loss used in ordinary GANs.

## Usage

Save two different datasets into data/ (I used SYNTHAI video sequences from http://synthia-dataset.net/downloads/ (CVPR16)
SYNTHIA-SEQS-01-NIGHT and SYNTHIA-SEQS-01-SUMMER). Then run main.py using
python3, setting any flags defined here:

```python
--technique (default MUNIT)
--debugging (default False)
--batch_size (default 1)
--epochs (default 200)
--run_name (default None)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
Distributed under the MIT License. See `LICENSE` for more information.
