#!/usr/bin/env python3

"""
PySpectrometer2 Les Wright 2022
https://www.youtube.com/leslaboratory
https://github.com/leswright1977

This project is a follow on from: https://github.com/leswright1977/PySpectrometer

This is a more advanced, but more flexible version of the original program. Tk Has been dropped as the GUI to allow fullscreen mode on Raspberry Pi systems and the iterface is designed to fit 800*480 screens, which seem to be a common resolutin for RPi LCD's, paving the way for the creation of a stand alone benchtop instrument.

Whats new:
Higher resolution (800px wide graph)
3 row pixel averaging of sensor data
Fullscreen option for the Spectrometer graph
3rd order polymonial fit of calibration data for accurate measurement.
Improved graph labelling
Labelled measurement cursors
Optional waterfall display for recording spectra changes over time.
Key Bindings for all operations

All old features have been kept, including peak hold, peak detect, Savitsky Golay filter, and the ability to save graphs as png and data as CSV.

For instructions please consult the readme!

--- MODIFICADO ---
Se reemplazo la clase GStreamerCamera (que leia el pipe crudo de gst-launch-1.0
via subprocess y hacia un reshape manual) por cv2.VideoCapture + appsink.

Motivo: nvvidconv en Jetson puede alinear (pad) cada fila del buffer a un
multiplo de bytes distinto de width*4. El reshape manual asumia stride =
width*4 exacto, y cuando el stride real tenia relleno extra, ese relleno se
interpretaba como pixeles, provocando que un fragmento del lado derecho de
cada fila apareciera "enrollado" al lado izquierdo de la fila siguiente
(el efecto de corte/repeticion en el borde izquierdo que se ve en pantalla).
appsink delega el manejo del stride a GStreamer/OpenCV, evitando el bug de raiz.
"""

import cv2
import time
import threading
import numpy as np
from specFunctions import (
    wavelength_to_rgb,
    savitzky_golay,
    peakIndexes,
    readcal,
    writecal,
    background,
    generateGraticule,
)
import base64
import argparse

# --- API: dependencies to expose teh video via http ---
import uvicorn
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse

# --- API: shared state between the OpenCV loop and the FastAPI server ---
api_app = FastAPI()
api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
frame_lock = threading.Lock()
latest_jpeg = {"spectrum": None, "waterfall": None}

def encode_and_store(name, img, quality=80):
    try:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    except Exception as e:
        print(f"[API-DEBUG] EXCEPCION codificando '{name}':{e}")
        return
    if ok:
        with frame_lock:
            latest_jpeg[name] = buf.tobytes()
 
def mjpeg_generator(name):
    boundary = b"--frame\r\n"
    while True:
        with frame_lock:
            frame = latest_jpeg.get(name)
        if frame is None:
            time.sleep(0.05)
            continue
        yield (boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(1/30) # -30fps output cap, independet of capture fps

@api_app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
    <head><title>PySpectrometer 2 - Live</title></head>
    <body style="margin:0; background:#000; text-align:center;">
        <h3 style="color:#0f0; font-family:sans-serif;">Spectrograph</h3>
        <img src="/video_feed" style="max-width:100%" />
        <h3 style="color:#0f0; font-family:sans-serif;">Waterfall</h3>
        <img src="/video_feed_waterfall" style="max-width:100%" />
    </body>
</html>
"""

@api_app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator("spectrum"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@api_app.get("/video_feed_waterfall")
def video_feed_waterfall():
    return StreamingResponse(
        mjpeg_generator("waterfall"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

@api_app.get("/snapshot.jpg")
def snapshot_jpg():
    with frame_lock:
        frame = latest_jpeg.get("spectrum")
    if frame is None:
        return Response(status_code=503)
    return Response(content=frame, media_type="image/jpeg")

def start_api_server(host="0.0.0.0", port=8000):
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    uvicorn.run(api_app, host=host, port=port, log_level="warning")

parser = argparse.ArgumentParser()
parser.add_argument(
    "--device",
    type=int,
    default=0,
    help="Video Device number e.g. 0, use v4l2-ctl --list-devices",
)
parser.add_argument("--fps", type=int, default=15, help="Frame Rate e.g. 30")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--fullscreen", help="Fullscreen (Native 800*480)", action="store_true"
)
group.add_argument(
    "--waterfall", help="Enable Waterfall (Windowed only)", action="store_true"
)
args = parser.parse_args()
dispFullscreen = False
dispWaterfall = False
if args.fullscreen:
    print("Fullscreen Spectrometer enabled")
    dispFullscreen = True
if args.waterfall:
    print("Waterfall display enabled")
    dispWaterfall = True

if args.device:
    dev = args.device
else:
    dev = 0

if args.fps:
    fps = args.fps
else:
    fps = 15

frameWidth = 800
frameHeight = 600


def build_gst_pipeline(width, height, fps, sensor_id=0, flip_method=0):
    """
    Pipeline GStreamer que termina en appsink en vez de filesink+stdout.
    OpenCV (y GStreamer) se encargan del stride internamente, por lo que
    no hace falta reshape manual ni riesgo de desalineamiento de filas.
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){width}, height=(int){height}, "
        f"framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true sync=false max-buffers=1"
    )


print("[Info] Iniciando video via GStreamer/appsink para la IMX477...")
gst_str = build_gst_pipeline(frameWidth, frameHeight, fps)
cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("ERROR CRITICO: No se pudo abrir el stream de GStreamer.")
    exit()

# --- API: Start FastAPI sever in background ---

api_thread = threading.Thread(target=start_api_server, kwargs={"port": 8000}, daemon=True)
api_thread.start()
print("[info] API de video disponible en http://0.0.0.0:8000/video_feed")

print("[info] W, H, FPS")
print(frameWidth)
print(frameHeight)
print(fps)


title1 = "PySpectrometer 2 - Spectrograph"
title2 = "PySpectrometer 2 - Waterfall"
stackHeight = (
    320 + 80 + 80
)  # height of the displayed CV window (graph+preview+messages)

if dispWaterfall == True:
    # watefall first so spectrum is on top
    cv2.namedWindow(title2, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(title2, frameWidth, stackHeight)
    cv2.moveWindow(title2, 200, 200)

if dispFullscreen == True:
    cv2.namedWindow(title1, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(title1, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
else:
    cv2.namedWindow(title1, cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(title1, frameWidth, stackHeight)
    cv2.moveWindow(title1, 0, 0)

# settings for peak detect
savpoly = 7  # savgol filter polynomial max val 15
mindist = 50  # minumum distance between peaks max val 100
thresh = 20  # Threshold max val 100

calibrate = False

clickArray = []
cursorX = 0
cursorY = 0


def handle_mouse(event, x, y, flags, param):
    global clickArray
    global cursorX
    global cursorY
    mouseYOffset = 160
    if event == cv2.EVENT_MOUSEMOVE:
        cursorX = x
        cursorY = y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX = x
        mouseY = y - mouseYOffset
        clickArray.append([mouseX, mouseY])


# listen for click on plot window
cv2.setMouseCallback(title1, handle_mouse)


font = cv2.FONT_HERSHEY_SIMPLEX

intensity = [0] * frameWidth  # array for intensity data...full of zeroes

holdpeaks = False  # are we holding peaks?
measure = False  # are we measuring?
recPixels = False  # are we measuring pixels and recording clicks?


# messages
msg1 = ""
saveMsg = "No data saved"

# blank image for Waterfall
waterfall = np.zeros([320, frameWidth, 3], dtype=np.uint8)
waterfall.fill(0)  # fill black

# Go grab the computed calibration data
caldata = readcal(frameWidth)
wavelengthData = caldata[0]
calmsg1 = caldata[1]
calmsg2 = caldata[2]
calmsg3 = caldata[3]

# generate the craticule data
graticuleData = generateGraticule(wavelengthData)
tens = graticuleData[0]
fifties = graticuleData[1]


def snapshot(savedata):
    now = time.strftime("%Y%m%d--%H%M%S")
    timenow = time.strftime("%H:%M:%S")
    imdata1 = savedata[0]
    graphdata = savedata[1]
    if dispWaterfall == True:
        imdata2 = savedata[2]
        cv2.imwrite("waterfall-" + now + ".png", imdata2)
    cv2.imwrite("spectrum-" + now + ".png", imdata1)
    f = open("Spectrum-" + now + ".csv", "w")
    f.write("Wavelength,Intensity\r\n")
    for x in zip(graphdata[0], graphdata[1]):
        f.write(str(x[0]) + "," + str(x[1]) + "\r\n")
    f.close()
    message = "Last Save: " + timenow
    return message


while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret == True:
        y = int((frameHeight / 2) - 40)  # origin of the vertical crop
        x = 0  # origin of the horiz crop
        h = 80  # height of the crop
        w = frameWidth  # width of the crop
        cropped = frame[y : y + h, x : x + w]
        bwimage = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        rows, cols = bwimage.shape
        halfway = int(rows / 2)
        # show our line on the original image
        # now a 3px wide region
        cv2.line(
            cropped, (0, halfway - 2), (frameWidth, halfway - 2), (255, 255, 255), 1
        )
        cv2.line(
            cropped, (0, halfway + 2), (frameWidth, halfway + 2), (255, 255, 255), 1
        )

        # banner image
        decoded_data = base64.b64decode(background)
        np_data = np.frombuffer(decoded_data, np.uint8)
        img = cv2.imdecode(np_data, 3)
        messages = img

        # blank image for Graph
        graph = np.zeros([320, frameWidth, 3], dtype=np.uint8)
        graph.fill(255)  # fill white

        # Display a graticule calibrated with cal data
        textoffset = 12
        for position in tens:
            cv2.line(graph, (position, 15), (position, 320), (200, 200, 200), 1)

        for positiondata in fifties:
            cv2.line(graph, (positiondata[0], 15), (positiondata[0], 320), (0, 0, 0), 1)
            cv2.putText(
                graph,
                str(positiondata[1]) + "nm",
                (positiondata[0] - textoffset, 12),
                font,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        for i in range(320):
            if i >= 64:
                if i % 64 == 0:  # suppress the first line then draw the rest...
                    cv2.line(graph, (0, i), (frameWidth, i), (100, 100, 100), 1)

        # Now process the intensity data and display it
        for i in range(cols):
            # average the data of 3 rows of pixels:
            dataminus1 = bwimage[halfway - 1, i]
            datazero = bwimage[halfway, i]  # pull the pixel data from the halfway mark
            dataplus1 = bwimage[halfway + 1, i]
            data = (int(dataminus1) + int(datazero) + int(dataplus1)) / 3
            data = np.uint8(data)

            if holdpeaks == True:
                if data > intensity[i]:
                    intensity[i] = data
            else:
                intensity[i] = data

        if dispWaterfall == True:
            wdata = np.zeros([1, frameWidth, 3], dtype=np.uint8)
            index = 0
            for i in intensity:
                rgb = wavelength_to_rgb(round(wavelengthData[index]))
                luminosity = intensity[index] / 255
                b = int(round(rgb[0] * luminosity))
                g = int(round(rgb[1] * luminosity))
                r = int(round(rgb[2] * luminosity))
                wdata[0, index] = (r, g, b)
                index += 1
            waterfall = np.insert(waterfall, 0, wdata, axis=0)
            waterfall = waterfall[:-1].copy()

            hsv = cv2.cvtColor(waterfall, cv2.COLOR_BGR2HSV)

        if holdpeaks == False:
            intensity = savitzky_golay(intensity, 17, savpoly)
            intensity = np.array(intensity)
            intensity = intensity.astype(int)
            holdmsg = "Holdpeaks OFF"
        else:
            holdmsg = "Holdpeaks ON"

        index = 0
        for i in intensity:
            rgb = wavelength_to_rgb(round(wavelengthData[index]))
            r = rgb[0]
            g = rgb[1]
            b = rgb[2]
            cv2.line(graph, (index, 320), (index, 320 - i), (b, g, r), 1)
            cv2.line(graph, (index, 319 - i), (index, 320 - i), (0, 0, 0), 1, cv2.LINE_AA)
            index += 1

        # find peaks and label them
        textoffset = 12
        thresh = int(thresh)
        indexes = peakIndexes(intensity, thres=thresh / max(intensity), min_dist=mindist)
        for i in indexes:
            height = intensity[i]
            height = 310 - height
            wavelength = round(wavelengthData[i], 1)
            cv2.rectangle(
                graph,
                ((i - textoffset) - 2, height),
                ((i - textoffset) + 60, height - 15),
                (0, 255, 255),
                -1,
            )
            cv2.rectangle(
                graph,
                ((i - textoffset) - 2, height),
                ((i - textoffset) + 60, height - 15),
                (0, 0, 0),
                1,
            )
            cv2.putText(
                graph,
                str(wavelength) + "nm",
                (i - textoffset, height - 3),
                font,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.line(graph, (i, height), (i, height + 10), (0, 0, 0), 1)

        if measure == True:
            cv2.line(graph, (cursorX, cursorY - 140), (cursorX, cursorY - 180), (0, 0, 0), 1)
            cv2.line(
                graph, (cursorX - 20, cursorY - 160), (cursorX + 20, cursorY - 160), (0, 0, 0), 1
            )
            cv2.putText(
                graph,
                str(round(wavelengthData[cursorX], 2)) + "nm",
                (cursorX + 5, cursorY - 165),
                font,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        if recPixels == True:
            cv2.line(graph, (cursorX, cursorY - 140), (cursorX, cursorY - 180), (0, 0, 0), 1)
            cv2.line(
                graph, (cursorX - 20, cursorY - 160), (cursorX + 20, cursorY - 160), (0, 0, 0), 1
            )
            cv2.putText(
                graph,
                str(cursorX) + "px",
                (cursorX + 5, cursorY - 165),
                font,
                0.4,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
        else:
            clickArray = []

        if clickArray:
            for data in clickArray:
                mouseX = data[0]
                mouseY = data[1]
                cv2.circle(graph, (mouseX, mouseY), 5, (0, 0, 0), -1)
                cv2.putText(
                    graph,
                    str(mouseX),
                    (mouseX + 5, mouseY),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 0, 0),
                )

        # stack the images and display the spectrum
        spectrum_vertical = np.vstack((messages, cropped, graph))
        cv2.line(spectrum_vertical, (0, 80), (frameWidth, 80), (255, 255, 255), 1)
        cv2.line(spectrum_vertical, (0, 160), (frameWidth, 160), (255, 255, 255), 1)
        cv2.putText(spectrum_vertical, calmsg1, (490, 15), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(spectrum_vertical, calmsg3, (490, 33), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(spectrum_vertical, "Framerate: " + str(fps), (490, 51), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(spectrum_vertical, saveMsg, (490, 69), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(spectrum_vertical, holdmsg, (640, 15), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(spectrum_vertical, "Savgol Filter: " + str(savpoly), (640, 33), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(spectrum_vertical, "Label Peak Width: " + str(mindist), (640, 51), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(spectrum_vertical, "Label Threshold: " + str(thresh), (640, 69), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(title1, spectrum_vertical)
        encode_and_store("spectrum", spectrum_vertical) #API: publish frame

        if dispWaterfall == True:
            waterfall_vertical = np.vstack((messages, cropped, waterfall))
            cv2.line(waterfall_vertical, (0, 80), (frameWidth, 80), (255, 255, 255), 1)
            cv2.line(waterfall_vertical, (0, 160), (frameWidth, 160), (255, 255, 255), 1)
            textoffset = 12

            for positiondata in fifties:
                for i in range(162, 480):
                    if i % 20 == 0:
                        cv2.line(waterfall_vertical, (positiondata[0], i), (positiondata[0], i + 1), (0, 0, 0), 2)
                        cv2.line(waterfall_vertical, (positiondata[0], i), (positiondata[0], i + 1), (255, 255, 255), 1)
                cv2.putText(waterfall_vertical, str(positiondata[1]) + "nm", (positiondata[0] - textoffset, 475), font, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(waterfall_vertical, str(positiondata[1]) + "nm", (positiondata[0] - textoffset, 475), font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(waterfall_vertical, calmsg1, (490, 15), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(waterfall_vertical, calmsg2, (490, 33), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(waterfall_vertical, calmsg3, (490, 51), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(waterfall_vertical, saveMsg, (490, 69), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(waterfall_vertical, holdmsg, (640, 15), font, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow(title2, waterfall_vertical)
            encode_and_store("waterfall", waterfall_vertical)

        

        keyPress = cv2.waitKey(1)
        if keyPress == ord("q"):
            break
        elif keyPress == ord("h"):
            holdpeaks = not holdpeaks
        elif keyPress == ord("s"):
            graphdata = []
            graphdata.append(wavelengthData)
            graphdata.append(intensity)
            if dispWaterfall == True:
                savedata = []
                savedata.append(spectrum_vertical)
                savedata.append(graphdata)
                savedata.append(waterfall_vertical)
            else:
                savedata = []
                savedata.append(spectrum_vertical)
                savedata.append(graphdata)
            saveMsg = snapshot(savedata)
        elif keyPress == ord("c"):
            calcomplete = writecal(clickArray)
            if calcomplete:
                caldata = readcal(frameWidth)
                wavelengthData = caldata[0]
                calmsg1 = caldata[1]
                calmsg2 = caldata[2]
                calmsg3 = caldata[3]
                graticuleData = generateGraticule(wavelengthData)
                tens = graticuleData[0]
                fifties = graticuleData[1]
        elif keyPress == ord("x"):
            clickArray = []
        elif keyPress == ord("m"):
            recPixels = False
            measure = not measure
        elif keyPress == ord("p"):
            measure = False
            recPixels = not recPixels
        elif keyPress == ord("o"):
            savpoly += 1
            if savpoly >= 15:
                savpoly = 15
        elif keyPress == ord("l"):
            savpoly -= 1
            if savpoly <= 0:
                savpoly = 0
        elif keyPress == ord("i"):
            mindist += 1
            if mindist >= 100:
                mindist = 100
        elif keyPress == ord("k"):
            mindist -= 1
            if mindist <= 0:
                mindist = 0
        elif keyPress == ord("u"):
            thresh += 1
            if thresh >= 100:
                thresh = 100
        elif keyPress == ord("j"):
            thresh -= 1
            if thresh <= 0:
                thresh = 0
    else:
        break


# Everything done, release the vid
cap.release()
cv2.destroyAllWindows()
