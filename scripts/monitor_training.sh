#!/bin/bash
# Monitor NER Training Progress
# Usage: ./scripts/monitor_training.sh

echo "=========================================="
echo "🔍 NER TRAINING MONITOR"
echo "=========================================="
echo ""

# Check if process is running
if ps aux | grep -q "[t]rain_ner_robust.py"; then
    echo "✅ Training process is RUNNING"
    echo ""
    
    # Show process info
    echo "📊 Process Info:"
    ps aux | grep "[t]rain_ner_robust.py" | awk '{printf "   PID: %s | CPU: %s%% | MEM: %s%% | Time: %s\n", $2, $3, $4, $10}'
    echo ""
    
    # Show latest log entries
    echo "📝 Latest Training Progress:"
    echo "=========================================="
    
    if [ -f "logs/training_output.log" ]; then
        tail -n 20 logs/training_output.log
    else
        # Find latest ner_training log
        LATEST_LOG=$(ls -t logs/ner_training_robust_*.log 2>/dev/null | head -n1)
        if [ -n "$LATEST_LOG" ]; then
            tail -n 20 "$LATEST_LOG"
        else
            echo "No log files found yet..."
        fi
    fi
    
    echo ""
    echo "=========================================="
    echo "💡 Commands:"
    echo "   Watch live: tail -f logs/training_output.log"
    echo "   Kill training: pkill -f train_ner_robust.py"
    echo "   Check again: ./scripts/monitor_training.sh"
    
else
    echo "⚠️  Training process NOT running"
    echo ""
    
    # Check if it completed
    if [ -f "models/ner/ran_ner_model_robust/meta.json" ]; then
        echo "✅ Training appears to be COMPLETE!"
        echo "   Model found: models/ner/ran_ner_model_robust/"
        echo ""
        echo "📊 Check results:"
        echo "   cat models/ner/ner_metrics_robust.json"
    else
        echo "❌ Training may have failed or not started"
        echo ""
        echo "🔍 Check logs:"
        if [ -f "logs/training_output.log" ]; then
            echo "   Last 10 lines of training_output.log:"
            tail -n 10 logs/training_output.log
        else
            echo "   No output log found"
        fi
    fi
fi

echo ""
