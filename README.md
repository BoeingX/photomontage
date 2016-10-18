# Seamless Image Stitching

Seamless image stitching graph cut methods permitting to stitch multiple photos on a panorama, partially adapted from paper [Graphcut Textures: Image and Video Synthesis Using Graph Cut](http://www.cc.gatech.edu/cpl/projects/graphcuttextures/gc-final-lowres.pdf). 

# Dependencies

-   `cmake`
-   `opencv` 3.0

# Compilation

    mkdir build
    cd build
    cmake ..
    make

# Execution

    ./main img1 img2 ...

# Example

## Input Photos

![alt text](./examples/IMG_0034.JPG "Photo 1")
![alt text](./examples/IMG_0035.JPG "Photo 2")
![alt text](./examples/IMG_0036.JPG "Photo 3")
![alt text](./examples/IMG_0037.JPG "Photo 4")

## Output Photo

![alt text](./examples/output_entire.jpg "Output")
