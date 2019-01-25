import cv2
import sys


def main():
    path = sys.argv[1]
    #filename=sys.argv[2]
    im = cv2.imread(path, 0)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL)
    cv2.imshow('image',im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print im.shape
    

    


if __name__ == "__main__":
    main()