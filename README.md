SuperFlux
=========

Python reference implementation of the SuperFlux onset detection algorithm as
described in:

"Maximum Filter Vibrato Suppression for Onset Detection"
by Sebastian Böck and Gerhard Widmer.
Proceedings of the 16th International Conference on Digital Audio Effects
(DAFx-13), Maynooth, Ireland, September 2013.

and the additional local group delay (LGD) based weighting scheme described in:

"Local group delay based vibrato and tremolo suppression for onset detection"
by Sebastian Böck and Gerhard Widmer.
Proceedings of the 13th International Society for Music Information
Retrieval Conference (ISMIR), Curitiba, Brazil, November 2013.

The papers can be downloaded from: 

<http://phenicx.upf.edu/system/files/publications/Boeck_DAFx-13.pdf>

<http://phenicx.upf.edu/system/files/publications/Boeck_ISMIR_2013.pdf>

If you use this software, please cite the corresponding paper.

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

@inproceedings{Boeck2013a,
	Author = {B{\"o}ck, Sebastian and Widmer, Gerhard},
	Title = {Local Group Delay based Vibrato and Tremolo Suppression for Onset Detection},
	Booktitle = {{Proceedings of the 13th International Society for Music Information Retrieval Conference (ISMIR), 2013.},
	Pages = {589–-594},
	Address = {Curitiba, Brazil},
	Month = {November},
	Year = {2013}

```


Usage
-----
`SuperFlux.py input.wav` processes the audio file and writes the detected
onsets to a file named `input.superflux.txt`.

Please see the `-h` option to get a more detailed description of the available
options, e.g. changing the suffix for the detection files.

Requirements
------------
* Python 2.7
* Numpy
* Scipy

