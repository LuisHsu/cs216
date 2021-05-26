const FPS = 30;
function onOpenCVReady(){
    let video = document.getElementById("videoInput")
    navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        .then(function(stream) {
            video.srcObject = stream;
            video.play();
        })
        .catch(function(err) {
            console.log("An error occurred! " + err);
        });
    cv['onRuntimeInitialized']=()=>{
        let source = new cv.Mat(video.height, video.width, cv.CV_8UC4);
        let capture = new cv.VideoCapture(video);
        function processVideo() {
            let begin = Date.now();
            capture.read(source);
            cv.imshow("output", source);
            // schedule next one.
            setTimeout(processVideo, 1000/FPS - (Date.now() - begin));
        }
        // schedule first one.
        setTimeout(processVideo, 0);
    };
}
