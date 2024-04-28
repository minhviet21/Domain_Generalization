## This is the source code of Domain Generalization In Image Classification Project of the course IT4343E at HUST.

## Contributions

| No. | Name              | Student ID | Email                                  |
|-----|-------------------|------------|----------------------------------------|
| 1   | Doan Minh Viet    | 20210933   | viet.dm210933@sis.hust.edu.vn          |
| 2   | Nguyen Viet Trung | 20214934   | trung.nv214934@sis.hust.edu.vn         |
| 3   | Do Hoang Tuan     | 20214939   | tuan.dh214939@sis.hust.edu.vn          |
| 4   | Dau Van Can       | 20214879   | can.dv214879@sis.hust.edu.vn           |
| 5   | Nguyen Ba Duong   | 20214886   | duong.nb212886@sis.hust.edu.vn         |

## Train model ##
```
!git clone https://github.com/minhviet21/Domain_Generalization_For_Image_Classification
```
```
from google.colab import drive
drive.mount('/content/drive/')
```
```
%cd /content/Domain_Generalization_For_Image_Classification
!python train.py
```