//learn file name, prepare file and Fiji for analysis
name=File.nameWithoutExtension;
rename("A");
run("Rotate... ", "angle=180 grid=1 interpolation=Bilinear stack");
run("Z Project...", "projection=[Max Intensity]");
setTool("freehand");
run("Set Measurements...", "area mean min integrated display redirect=None decimal=3");


//Close unnecessary windows from last analysis
if (isOpen("Results")) { 
         selectWindow("Results"); 
         run("Close"); 
    } 
if (isOpen("Summary")) { 
         selectWindow("Summary"); 
         run("Close"); 
    } 
if (isOpen("ROI Manager")) { 
         selectWindow("ROI Manager"); 
         run("Close"); 
    } 

//Optional: Input ROI File:
//roi=File.openDialog("Select ROI file");
//roiManager("Open",roi);

//Define ROIs Manually (background, experimental and control sides)
waitForUser("Draw Background ROI 1, then press ok");
roiManager("Add");
roiManager("Select",0);
roiManager("Rename","background");
roiManager("Show All");
waitForUser("Draw Background ROI 2, then press ok");
roiManager("Add");
roiManager("Select",1);
roiManager("Rename","background");
waitForUser("Draw Background ROI 3, then press ok");
roiManager("Add");
roiManager("Select",2);
roiManager("Rename","background");


waitForUser("Draw Experimental ROI, then press ok");
roiManager("Add");
roiManager("Select",3);
roiManager("Rename","Expt");
waitForUser("Draw Control ROI, then press ok");
roiManager("Add");
roiManager("Select",4);
roiManager("Rename","Cntl");
run("Split Channels");

//Measure background then ROI IntDen
//Channel 4
selectWindow("C4-MAX_A");
rename("Ceramide");
resetMinAndMax();
//run("Median...", "radius=3");
roiManager("Show All");
roiManager("Select", 0);
run("Measure");
roiManager("Select", 1);
run("Measure");
roiManager("Select", 2);
run("Measure");
roiManager("Select", 4);
run("Measure");
roiManager("Select", 3);
run("Measure");

//Save out ROIs
waitForUser("Choose a directory to save ROIs");
dir = getDirectory("Choose a directory to save ROI sets");
roiManager("Save", dir+name+".zip");

//Save out Measurements as csv
waitForUser("Choose a directory to save measurements");
dir = getDirectory("Choose a directory to save CSV measurement results");
saveAs("Results", dir+name+".csv");

//Close image windows
run("Close All");