package team6072.vision;

import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.*;
import edu.wpi.first.networktables.*;

public class GPBasicListener implements VisionRunner.Listener<GripPipelineBasic> {

    // The object to synchronize on to make sure the vision thread doesn't
    // write to variables the main thread is using.
    private final Object visionLock = new Object();

    private NetworkTable mTbl;

    public GPBasicListener() {
        NetworkTableInstance tblInst = NetworkTableInstance.getDefault();
        mTbl = tblInst.getTable("vision");
    }

    private int mCallCounter = 0;

    /**
     * Called when the pipeline has run. We need to grab the output from the
     * pipeline then communicate to the rest of the system over network tables
     */
    @Override
    public void copyPipelineOutputs(GripPipelineBasic pipeline) {
        synchronized (visionLock) {
            // Take a snapshot of the pipeline's output because it may have changed the next
            // time this method is called!
            mTbl.getEntry("GPBasicKey").setString("Call: " + mCallCounter);
            NetworkTableEntry f = mTbl.getEntry("KeyKey");
            f.forceSetDouble(55);
            mCallCounter++;
        }
    }

}