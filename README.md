# Seamless Image Stitching

Seamless image stitching by graph cut method, partially adapted from paper [Graphcut Textures: Image and Video Synthesis Using Graph Cut](http://www.cc.gatech.edu/cpl/projects/graphcuttextures/gc-final-lowres.pdf). 

# Dependencies

-   `cmake`
-   `opencv` 3.0
-   [`maxflow`](http://pub.ist.ac.at/~vnk/software.html)

# Compilation

    mkdir build
    cd build
    cmake ..
    make

# Execution

    ./main img1 img2 ...

>   Input images are not necessarily in good order in case of panorama.

# Example

## Panorama

### Input Photos

![alt text](./examples/IMG_0034.JPG "Photo 1")
![alt text](./examples/IMG_0035.JPG "Photo 2")
![alt text](./examples/IMG_0036.JPG "Photo 3")
![alt text](./examples/IMG_0037.JPG "Photo 4")

### Output Photo

![alt text](./examples/output_entire.jpg "Output")

## Image Synthese

### Input Photo

![alt text](./examples/cherries.jpg "Cherries")

### Output Photo

![alt text](./examples/cherriesx2.jpg "Cherries x 2")
