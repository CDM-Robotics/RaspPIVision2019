package team6072.vision;

import edu.wpi.first.networktables.NetworkTableInstance;
import edu.wpi.first.vision.*;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;

import edu.wpi.cscore.CvSource;
import edu.wpi.cscore.VideoSource;
import edu.wpi.first.cameraserver.CameraServer;
import edu.wpi.first.networktables.*;
import org.opencv.core.MatOfPoint2f;

import org.opencv.imgproc.Imgproc;

import org.opencv.core.Point;
import java.util.ArrayList;
import team6072.vision.PIPipeline.Line;

import java.util.List;

import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;

public class PIPipelineListener implements VisionRunner.Listener<PIPipeline> {

    // The object to synchronize on to make sure the vision thread doesn't
    // write to variables the main thread is using.
    private final Object visionLock = new Object();

    private NetworkTable mTbl;
    private NetworkTableEntry mX1;
    private NetworkTableEntry mY1;
    private NetworkTableEntry mX2;
    private NetworkTableEntry mY2;   
    
    private CameraServer mCameraServer;
    private CvSource mMaskOutput;
    private CvSource mOpenCVSource;

    
    private int mCallCounter = 0;
    private int mCounter;

    public PIPipelineListener() {
        NetworkTableInstance tblInst = NetworkTableInstance.getDefault();
        mTbl = tblInst.getTable("vision");  
        mCameraServer = CameraServer.getInstance();
        mMaskOutput = mCameraServer.putVideo("Mask Output", 320, 240);
        mOpenCVSource = mCameraServer.putVideo("OpenCV Source", 320, 240);
    }

     
    /**
     * Called when the pipeline has run. We need to grab the output from the
     * pipeline then communicate to the rest of the system over network tables
     */
    @Override
    public void copyPipelineOutputs(PIPipeline pipeline) {
        synchronized (visionLock) {
            mTbl.getEntry("PIKey").setString("Call: " + mCounter);
            mCallCounter++;


            if(mCallCounter - 1 == mCounter){
                PIPipeline pi = pipeline;
                ArrayList<Line> lines = pi.findLinesOutput();
                mMaskOutput.putFrame(pi.maskOutput());
                ArrayList<MatOfPoint> contours = pi.findContoursOutput();
                ArrayList<Rect> rectangles = new ArrayList<Rect>();

                if(pi.findContoursOutput().size() > 0){
                    for (int x = 0; x < contours.size(); x++) {
                        Rect r = Imgproc.boundingRect(contours.get(x));
                        rectangles.add(r);
                        Imgproc.rectangle(pi.getImage0(), new Point(r.x, r.y), new Point(r.x+r.width, r.y+r.height), new Scalar(255 , 255, 255), 5);
                    } //end for loop 
                }
                // Take a snapshot of the pipeline's output because it may have changed the next
                // time this method is called!
                if(lines != null) {
                    if(lines.size() > 0)
                    {
                        int i = 0;
                        mX1 = mTbl.getEntry("line : " + i + " x1");
                        mY1 = mTbl.getEntry("line : " + i + " y1");
                        mX2 = mTbl.getEntry("line : " + i + " x2");
                        mY2 = mTbl.getEntry("line : " + i + " y2");  
                        mX1.setDouble(lines.get(i).x1);
                        mY1.setDouble(lines.get(i).y1);
                        mX2.setDouble(lines.get(i).x2);
                        mY2.setDouble(lines.get(i).y2);                        
                    }
                    // for(int i = 0; i < lines.size(); i ++)
                    // {
                    //     mX1 = mTbl.getEntry("line : " + i + " x1");
                    //     mY1 = mTbl.getEntry("line : " + i + " y1");
                    //     mX2 = mTbl.getEntry("line : " + i + " x2");
                    //     mY2 = mTbl.getEntry("line : " + i + " y2");  
                    //     mX1.setDouble(lines.get(i).x1);
                    //     mY1.setDouble(lines.get(i).y1);
                    //     mX2.setDouble(lines.get(i).x2);
                    //     mY2.setDouble(lines.get(i).y2);
                    // }
                }           
                mCounter++;
            }else{
                mCallCounter--;
            }
        }
    }

}