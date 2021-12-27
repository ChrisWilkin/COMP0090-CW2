# This is the original, unprocessed, dataset
Contains labels for binary classification, image segmentation, bounding boxes, and breed labels

AllData.h5 Structure--------------------------------
/train
    /images
        /images
            Image Data  256x256x3
        /ID 
            ID values linked to the images
    /masks
        /masks
            Masks Data  256x256, 2/1 = background, 0 = foreground
        /ID 
            ID values linked to the images
    /bboxes
        /bboxes
            Bounding Box Data [xmin, ymin, xmax, ymax]
        /ID 
            ID values linking BB data
    /bins
        /bins
            Binary Classification data : (0/1), (0-37) --> First number 0/1 Cat/Dog, Second number 0-37 breed
        /ID
            ID values
/test
    /images
        /images
            Image Data  256x256x3
        /ID 
            ID values linked to the images
    /masks
        /masks
            Masks Data  256x256, 2/1 = background, 0 = foreground
        /ID 
            ID values linked to the images
    /bboxes
        /bboxes
            Bounding Box Data [xmin, ymin, xmax, ymax]
        /ID 
            ID values linking BB data
    /bins
        /bins
            Binary Classification data : (0/1), (0-37) --> First number 0/1 Cat/Dog, Second number 0-37 breed
        /ID
            ID values
/val
    /images
        /images
            Image Data  256x256x3
        /ID 
            ID values linked to the images
    /masks
        /masks
            Masks Data  256x256, 2/1 = background, 0 = foreground
        /ID 
            ID values linked to the images
    /bboxes
        /bboxes
            Bounding Box Data [xmin, ymin, xmax, ymax]
        /ID 
            ID values linking BB data
    /bins
        /bins
            Binary Classification data : (0/1), (0-37) --> First number 0/1 Cat/Dog, Second number 0-37 breed
        /ID
            ID values