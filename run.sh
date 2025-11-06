if lsmod | grep -q $GALCORE_DRIVER_NAME; then
        echo "Driver module $GALCORE_DRIVER_NAME is loaded, uninstalling..."
        rmmod galcore
        echo "old galcore rmmod"
fi
sleep 3

insmod  /root/drivers/galcore.ko  registerMemBase=0xf8800000 contiguousSize=0x200000 irqLine=21 showArgs=1 recovery=1 stuckDump=2 powerManagement=1
./main