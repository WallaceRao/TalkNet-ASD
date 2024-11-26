
COMMAND='python demoTalkNet.py'
LOGFILE=taklnet.log

writelog() {
  now=`date`
  echo "$now $*" >> $LOGFILE
}

writelog "Starting"
while true ; do
  $COMMAND
  writelog "Exited with status $?"
  sleep 10
  writelog "Restarting"
done
