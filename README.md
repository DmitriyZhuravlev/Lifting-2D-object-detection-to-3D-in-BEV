# Lifting 2D Object Detection to 3D: Geometric Approach in Bird-Eye-View





<div align="center">
<p>
<img src="img/Pexels Videos 2053100_720.gif" width="800"/> 
</p>
<br>

 
</div>

</div>


## Introduction

This repository contains PoC implementation of the https://link.springer.com/chapter/10.1007/978-3-031-09076-9_21.
3D detection works on the top of https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.


## To run the 3D tracker

1. git clone --recurse-submodules git@github.com:DmitriyZhuravlev/Lifting-2D-object-detection-to-3D-in-BEV.git
2. pip install -r requirements.txt
3. python track_3d.py --classes 2 3 7 --conf-thres 0.7


## Cite

If you find this project useful in your research, please consider cite:

```latex
@misc{yolov5-strongsort-osnet-2022,
    title={Lifting-2D-object-detection-to-3D-in-BEV},
    author={Dmitriy Zhuravlev},
    howpublished = {\url{https://github.com/DmitriyZhuravlev/Lifting-2D-object-detection-to-3D-in-BEV}},
    year={2022}
}
```

## Contact 

For bugs and feature requests please visit [GitHub Issues](https://github.com/DmitriyZhuravlev/Lifting-2D-object-detection-to-3D-in-BEV/issues). For business inquiries or professional support requests please send an email to: dzhuravlev@ukr.net
