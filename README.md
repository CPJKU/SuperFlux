SuperFlux
=========

Python reference implementation of the SuperFlux onset detection algorithm described in:

"Maximum Filter Vibrato Suppression for Onset Detection"
by Sebastian Böck and Gerhard Widmer
in Proceedings of the 16th International Conference on Digital Audio Effects
(DAFx-13), Maynooth, Ireland, September 2013

The paper can be downloaded from <http://phenicx.upf.edu/system/files/publications/Boeck_DAFx-13.pdf>.

If you use this software, please cite the paper.

```
@inproceedings{Boeck2013,
	Author = {B{\"o}ck, Sebastian and Widmer, Gerhard},
	Title = {Maximum Filter Vibrato Suppression for Onset Detection},
	Booktitle = {{Proceedings of the 16th International Conference on Digital Audio Effects (DAFx-13)}},
	Pages = {55--61},
	Address = {Maynooth, Ireland},
	Month = {September},
	Year = {2013}
}

```


Usage
-----
`SuperFlux.py input.wav` processes the audio file and writes the detected onsets to a file named input.onsets.txt.

Please see the `-h` option to get a more detailed description of the available options.

Requirements
------------
* Python 2.7
* Numpy
* Scipy

