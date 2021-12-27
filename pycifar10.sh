GPUID=0
OUTDIR=results/resnet18
REPEAT=1
mkdir -p $OUTDIR
BATCH_SIZE=4096
EPOCHS=1000
LEARNERPATH=/home/hikmat/Desktop/JWorkspace/CL/PYCIFAR10
python -u $LEARNERPATH/main.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam --epochs $EPOCHS --batch_size $BATCH_SIZE --model_name resnet18 --lr 0.01            | tee ${OUTDIR}/resnet18.log

