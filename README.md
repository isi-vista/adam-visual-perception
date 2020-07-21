# ADAM Visual Perception

This repository explores how two aspects of visual perception
which are vital for early language learning
can be captured by algorithms.
It consists of two sub-projects:

- [Motion cause detection](doc/motion_cause_detection.md)
- [Gaze object detection](doc/gaze_object_detection.md)

The library code is shared for the two sub-projects. Below we present the guide on how to install and run it locally or on a docker image, as well as how to get the required data. 

Click on the sub-project documentation link above to read more about the sub-project setting and see the step-by-step guide on how to prepare and run experiments.

This code was developed by [Mikayel Samvelyan](https://github.com/samvelyan) under the direction of [Ryan Gabbard](https://github.com/gabbard) and [Marjorie Freedman](https://www.isi.edu/people/mrf/about) as part of [the Information Science Institute's DARPA GAILA research effort on algorithmic models of child language learning](https://github.com/isi-vista/adam).  Should you have any question, please reach out to [Mikayel Samvelyan](mailto:mikayel@samvelyan.com) and [Marjorie Freedman](mailto:mrf@isi.edu).

## Installation

See [install.md](doc/install.md).

## Data

We have gathered a large number of videos of educational children's television series, such as [Mister Rogers' Neighborhood](https://en.wikipedia.org/wiki/Mister_Rogers%27_Neighborhood) and [Sesame Street](https://en.wikipedia.org/wiki/Sesame_Street), and created the initial version of the benchmark.

The video files are downloaded from [Internet Archive](https://archive.org/) which is a non-profit library of millions of free books, movies, and more. Here are the links:

- Neighborhood (50 episodes, 8.2 GB) - [Link](https://archive.org/details/MisterRogersNeighborhoodEpisodes), [Download Link](https://archive.org/compress/MisterRogersNeighborhoodEpisodes/formats=MATROSKA&file=/MisterRogersNeighborhoodEpisodes.zip)
- Sesame Street (Episode 3037, 444 MB) - [Link](https://archive.org/details/sesame3037), [Download Link](https://ia801006.us.archive.org/1/items/sesame3037/3037.mp4)
- Sesame Street (Episode 2257, 567 MB) - [Link](https://archive.org/details/SesameStreet2257), [Download Link](https://ia803009.us.archive.org/31/items/SesameStreet2257/Sesame-Street-_2257_.mp4)
- Sesame Street (Episode 2517, 846 MB) - [Link](https://archive.org/details/sesamestreetepisode2517convertvideoonline.com), [Download Link](https://ia803107.us.archive.org/11/items/sesamestreetepisode2517convertvideoonline.com/Sesame%20Street%20%28Episode%202517%29%20%28convert-video-online.com%29.mp4)

### Metadata

The information about the metadata on video segments can be found in the `benchmark` directory. 

- `motion_raw.tsv` stores information on videos for the **motion** sub-project prior to preprocessing. `motion.tsv` contains information on video segments after the preprocessing.

- `gaze_raw.tsv` stores information on videos for the **gaze** sub-project prior to preprocessing. `gaze.tsv` contains information on video segments after the preprocessing.

### Video segments

Preprocessed video segments can be found in the `data` directory.

- `videos_motion` directory contains the preprocessed video segments for the **motion** sub-project.

- `videos_gaze` directory contains the preprocessed video segments for the **gaze** sub-project.
