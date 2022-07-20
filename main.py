# -*- coding: utf8 -*-
import numpy as np
import matplotlib.pyplot  as plt
import cv2


def perspect_warp(im: np.ndarray, anglex: float = 0, angley: float = 0, anglez: float = 0):
    '''
    input: 
        im: 
        anglex: pitch angle
        angley: yaw angle
    output:

    '''
    pad = 50
    im1 = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, 0)
    h, w = im1.shape[:2]
    z = np.sqrt(w ** 2 + h ** 2)
    rx = np.array([[1, 0, 0, 0],
                    [0, np.cos(np.radians(anglex)), -np.sin(np.radians(anglex)), 0],
                    [0, np.sin(np.radians(anglex)), np.cos(np.radians(anglex)), 0, ],
                    [0, 0, 0, 1]], np.float32)

    ry = np.array([[np.cos(np.radians(angley)), 0, np.sin(np.radians(angley)), 0],
                    [0, 1, 0, 0],
                    [-np.sin(np.radians(angley)), 0, np.cos(np.radians(angley)), 0],
                    [0, 0, 0, 1]], np.float32)

    rz = np.array([[np.cos(np.radians(anglez)), -np.sin(np.radians(anglez)), 0, 0],
                    [np.sin(np.radians(anglez)), np.cos(np.radians(anglez)), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], np.float32)

    r = rx @ ry @ rz

    pcenter = np.array([w / 2, h / 2, 0, 0], np.float32)

    ps = np.array([[0, 0, 0, 0], [w, 0, 0, 0], [0, h, 0, 0], [w, h, 0, 0]], dtype = np.float32) - pcenter

    dst = (r @ ps.T).T

    org = np.array([[0, 0],
                    [w, 0],
                    [0, h],
                    [w, h]], np.float32)

    dst1 = dst[:, :2] * z / (dst[:, 2:3] + z) + pcenter[:2]

    warpR = cv2.getPerspectiveTransform(org, dst1)
    im2 = cv2.warpPerspective(im1, warpR, (w, h))
    return im2


def main():
    '''
    '''
    im = cv2.imread("little tiger.jpg")
    
    anglex = 0
    angley = 30
    anglez = 0
    while 1:
        cv2.imshow("result", perspect_warp(im, anglex, angley, anglez))
        c = cv2.waitKey(30)

        if c == ord('w'):
            anglex += 1
        if c == ord('s'):
            anglex -= 1
        if c == ord('a'):
            angley += 1
        if c == ord('d'):
            angley -= 1
        if c == ord('u'):
            anglez += 1
        if c == ord('p'):
            anglez -= 1
        if c == ord(' '):
            anglex = angley = anglez = 0
        
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()
