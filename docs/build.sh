#/bin/bash

# Use package from parent directory.
export PYTHONPATH=$(realpath ..):$PYTHONPATH

REPORTDIR=_build/html/reports
jupyter-book build -W --keep-going .
RETVAL=$?
if [ $RETVAL -ne 0 ]; then
    if [ -e $REPORTDIR ]; then
      echo "Error occured; showing saved reports"
      cat $REPORTDIR/*
    fi
else
    # Clear out any old reports
    rm -f $REPORTDIR/*
fi
exit $RETVAL
