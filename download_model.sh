# SyncNet model and S3FD face detector (pre-processing pipeline)

mkdir -p data
curl -fL http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model -o data/syncnet_v2.model
curl -fL https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth -o data/sfd_face.pth
