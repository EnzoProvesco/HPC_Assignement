echo "Are you sure you want to delete all log files? (y/n)"
read -r confirm
if [ "$confirm" = "y" ]; then
    echo Deleting all files...
    rm -rf logData/*
    echo All files deleted.
else
    echo Operation cancelled.
fi
