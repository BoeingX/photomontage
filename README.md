# photomontage

Create a panorama with pairwise overlapped photos.

## First Version

Glue several **well-ordered** photos together.

### Pseudo-code

input: n **well-ordered** photos (n >= 1)

output = photo 0

for i = 1, 2, ..., n-1:
    
    calculate homography between output and photo i (c.f. TP5)

    apply homography transformation to photo i

    calculate the overlap between output and transformed photo i

    apply graph cut to determine a boundary

    update output

return output

##  Second Version

Do not require photos to be well-ordered


