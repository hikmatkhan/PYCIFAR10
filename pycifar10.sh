GPUID=0
OUTDIR=results/resnet18
REPEAT=1
#mkdir -p $OUTDIR
BATCH_SIZE=256
EPOCHS=50
#LEARNERPATH="/"
python -u main.py --outdir $OUTDIR --dataset CIFAR10 --train_aug --gpuid $GPUID --repeat $REPEAT --optimizer Adam --epochs $EPOCHS --batch_size $BATCH_SIZE --model_name resnet18 --lr 0.01            | tee ${OUTDIR}/resnet18.log

