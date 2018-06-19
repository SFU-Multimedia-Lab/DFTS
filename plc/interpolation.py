from scipy import interpolate
import numpy as np

def interpPackets(pBuffer, loss, InterpKind='linear'):
    #need to handle errors when kind is not implemented: NotImplementedError
    # print(loss)
    nz         = np.where(loss!=0)[0]
    nz.sort(kind='mergesort')
    # print(nz)
    z          = np.where(loss==0)[0]
    z.sort(kind='mergesort')
    # print(z)
    itpBuffer  = interpolate.interp1d(nz, pBuffer[nz], kind=InterpKind, axis=0, bounds_error=False)
    pBuffer[z] = itpBuffer(z)
    return pBuffer
