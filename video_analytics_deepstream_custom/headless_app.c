#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <cuda_runtime_api.h>

/* The Muxer Batch Size must match the number of sources (1) */
#define MUXER_OUTPUT_WIDTH 1920
#define MUXER_OUTPUT_HEIGHT 1080
#define MUXER_BATCH_TIMEOUT_USEC 40000

int main (int argc, char *argv[])
{
  GMainLoop *loop = NULL;
  GstElement *pipeline = NULL, *source = NULL, *h264parser = NULL,
      *decoder = NULL, *streammux = NULL, *sink = NULL, *pgie = NULL, *nvvidconv = NULL,
      *nvosd = NULL, *encoder = NULL, *codeparser = NULL, *qtmux = NULL, *filesink = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;

  /* Standard GStreamer Initialization */
  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* --- CREATE ELEMENTS --- */
  /* 1. Pipeline */
  pipeline = gst_pipeline_new ("ds-headless-pipeline");

  /* 2. Source (Read file) */
  source = gst_element_factory_make ("filesrc", "file-source");
  g_object_set (G_OBJECT (source), "location", "sample_720p.h264", NULL);

  /* 3. Parser & Decoder */
  h264parser = gst_element_factory_make ("h264parse", "h264-parser");
  decoder = gst_element_factory_make ("nvv4l2decoder", "nvv4l2-decoder");

  /* 4. Stream Muxer (Required by DeepStream to batch frames) */
  streammux = gst_element_factory_make ("nvstreammux", "stream-muxer");
  g_object_set (G_OBJECT (streammux), "batch-size", 1, NULL);
  g_object_set (G_OBJECT (streammux), "width", MUXER_OUTPUT_WIDTH, "height", MUXER_OUTPUT_HEIGHT,
      "batched-push-timeout", MUXER_BATCH_TIMEOUT_USEC, NULL);

  /* 5. The AI Model (nvinfer) */
  pgie = gst_element_factory_make ("nvinfer", "primary-nvinference-engine");
  /* Load the config file we copied earlier */
  g_object_set (G_OBJECT (pgie), "config-file-path", "dstest1_pgie_config.txt", NULL);

  /* 6. Visualization (Converter + OSD) */
  nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* 7. Encoding Pipeline (Video Convert -> Encode -> Parse -> MP4Mux -> FileSave) */
  GstElement *transform = gst_element_factory_make ("nvvideoconvert", "transform"); // Need convert for encoder
  encoder = gst_element_factory_make ("nvv4l2h264enc", "video-encoder");
  codeparser = gst_element_factory_make ("h264parse", "h264-parser-2");
  qtmux = gst_element_factory_make ("qtmux", "muxer");
  filesink = gst_element_factory_make ("filesink", "file-output");
  g_object_set (G_OBJECT (filesink), "location", "output_video.mp4", NULL);

  if (!pipeline || !source || !h264parser || !decoder || !streammux || !pgie ||
      !nvvidconv || !nvosd || !transform || !encoder || !codeparser || !qtmux || !filesink) {
    g_printerr ("One element could not be created. Exiting.\n");
    return -1;
  }

  /* --- BUILD PIPELINE --- */
  gst_bin_add_many (GST_BIN (pipeline),
      source, h264parser, decoder, streammux, pgie,
      nvvidconv, nvosd, transform, encoder, codeparser, qtmux, filesink, NULL);

  /* Link: Source -> Parser -> Decoder */
  gst_element_link_many (source, h264parser, decoder, NULL);

  /* Link: Decoder -> Muxer (Tricky: Request Pad) */
  GstPad *sinkpad, *srcpad;
  gchar pad_name_sink[] = "sink_0";
  sinkpad = gst_element_get_request_pad (streammux, pad_name_sink);
  srcpad = gst_element_get_static_pad (decoder, "src");
  if (gst_pad_link (srcpad, sinkpad) != GST_PAD_LINK_OK) {
      g_printerr ("Failed to link decoder to muxer.\n");
      return -1;
  }
  gst_object_unref (sinkpad);
  gst_object_unref (srcpad);

  /* Link: Muxer -> Inference -> Conv -> OSD -> Conv -> Encoder -> Parser -> Mux -> Sink */
  if (!gst_element_link_many (streammux, pgie, nvvidconv, nvosd, transform, encoder, codeparser, qtmux, filesink, NULL)) {
      g_printerr ("Elements could not be linked. Exiting.\n");
      return -1;
  }

  /* --- RUN --- */
  g_print ("🚀 Pipeline Initialized. Processing video... (Check 'output_video.mp4' when done)\n");
  gst_element_set_state (pipeline, GST_STATE_PLAYING);
  
  /* Wait until error or EOS (End Of Stream) */
  bus = gst_element_get_bus (pipeline);
  GstMessage *msg = gst_bus_timed_pop_filtered (bus, GST_CLOCK_TIME_NONE, GST_MESSAGE_ERROR | GST_MESSAGE_EOS);

  /* --- CLEANUP --- */
  if (msg != NULL) {
    switch (GST_MESSAGE_TYPE (msg)) {
      case GST_MESSAGE_ERROR: {
        GError *err;
        gchar *debug_info;
        gst_message_parse_error (msg, &err, &debug_info);
        g_printerr ("Error received from element %s: %s\n", GST_OBJECT_NAME (msg->src), err->message);
        g_printerr ("Debugging information: %s\n", debug_info ? debug_info : "none");
        g_clear_error (&err);
        g_free (debug_info);
        break;
      }
      case GST_MESSAGE_EOS:
        g_print ("✅ End-Of-Stream reached. Video processing complete.\n");
        break;
      default:
        g_printerr ("Unexpected message received.\n");
        break;
    }
    gst_message_unref (msg);
  }

  gst_object_unref (bus);
  gst_element_set_state (pipeline, GST_STATE_NULL);
  gst_object_unref (pipeline);
  return 0;
}
