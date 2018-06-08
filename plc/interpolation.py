from scipy import interpolate

def interpPackets(pBuffer, loss, InterpKind='linear'):
    #need to handle errors when kind is not implemented: NotImplementedError
    nz         = np.where(loss!=0)[0]
    z          = np.where(loss==0)[0]
    itpBuffer  = interpolate.interp1d(nz, pBuffer[nz], kind=InterpKind, axis=0)
    pBuffer[z] = itpTemp(z)
    return pBuffer
