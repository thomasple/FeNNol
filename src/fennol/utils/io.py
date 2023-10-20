import time
import numpy as np

def xyz_reader(filename,box_info=False,indexed=False
                  ,start=1, stop=-1,step=1,max_frames=None
                  ,stream=False,interval=2.,sleep=time.sleep):
  if stop>0 and start > stop: return
  pf=open(filename,"r")
  inside_frame=False
  nat_read=False
  iframe=0
  stride=step
  nframes=0
  if start>1 : print(f"skipping {start-1} frames")
  if not box_info: box=None
  while True:
    line=pf.readline()

    if not line:
      if stream:
        sleep(interval)
        continue
      elif inside_frame or nat_read:
        raise Exception("Error: premature end of file!")
      else:
        break
    
    line = line.strip()
    if line.startswith("#"): continue

    if not inside_frame:
      if not nat_read:
        nat=int(line.split()[0])
        nat_read=True
        iat=0
        inside_frame = not box_info
        xyz=np.zeros((nat,3))
        symbols=[]
        continue
      
      box=np.array(line.split(),dtype="float")
      inside_frame=True
      continue

    if not nat_read: raise Exception("Error: nat not read!")
    ls=line.split()
    iat,s = (int(ls[0]),1) if indexed else (iat+1,0)
    symbols.append(ls[s])
    xyz[iat-1,:] = np.array([ls[s+1],ls[s+2],ls[s+3]],dtype="float")
    if iat == nat:
      iframe+=1
      inside_frame=False
      nat_read=False
      if iframe < start: continue
      stride+=1
      if stride>=step:
        nframes+=1
        stride=0
        yield symbols,xyz,box
      if (max_frames is not None and nframes >= max_frames) \
          or (stop>0 and iframe >= stop):
        break

def read_xyz(filename,box_info=False,indexed=False
                  ,start=1, stop=-1,step=1,max_frames=None):
  return [ frame for frame 
              in xyz_reader(filename,box_info,indexed
                      ,start,stop,step,max_frames) 
          ]

def last_xyz_frame(filename,box_info=False,indexed=False):
  last_frame=None
  for frame in xyz_reader(filename,box_info,indexed
                      ,start=-1,stop=-1,step=1,max_frames=1):
    last_frame=frame
  return last_frame

def write_arc_frame(f,symbols,coordinates):
  nat=len(symbols)
  f.write(f'{nat}\n')
  #f.write(f'{axis} {axis} {axis} 90.0 90.0 90.0 \n')
  for i in range(nat):
    f.write(f'{i+1} {symbols[i]} {coordinates[i,0]} {coordinates[i,1]} {coordinates[i,2]}\n')
  f.flush()