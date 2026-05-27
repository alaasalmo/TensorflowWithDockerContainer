import os
import json
import socket
import threading
import time

import tensorflow as tf
from model import build_model

#1-Get parameters
DATA_PATH = os.environ.get("DATA_PATH", "/app/data/finance_news.csv")
OUTPUT_PATH_MODEL = os.environ.get("OUTPUT_PATH_MODEL", "/app/output/")
tf_config = json.loads(os.environ["TF_CONFIG"])
task_type = tf_config["task"]["type"]
task_index = tf_config["task"]["index"]

print("TF_CONFIG:", tf_config, flush=True)
print("Task:", task_type, task_index, flush=True)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 8))
EPOCHS = int(os.environ.get("EPOCHS", 5))
STEPS_PER_EPOCH = int(os.environ.get("STEPS_PER_EPOCH", 10))
VOCAB_SIZE = int(os.environ.get("VOCAB_SIZE", 5000))
MAX_LEN = int(os.environ.get("MAX_LEN", 50))


# 2- Socket Readiness Check
# Check for readiness for Chief, Worker-0 and worker-1. If the all workers are ready to start the process 

BASE_READINESS_PORT = int(os.environ.get("BASE_READINESS_PORT", 9000))
def start_readiness_server(port):
    def server():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("0.0.0.0", port))
        s.listen(5)

        print(f"[READINESS SERVER] Listening on port {port}", flush=True)

        while True:
            conn, addr = s.accept()
            conn.close()

    thread = threading.Thread(target=server, daemon=True)
    thread.start()


def get_readiness_port(task_type, task_index):
    if task_type == "chief":
        return BASE_READINESS_PORT

    return BASE_READINESS_PORT + task_index + 1

#3- Chief waits until workers are ready. Allow the chief to wait for 300 seconds. If there is stuck in the workers the run will fail
def wait_for_workers(tf_config, timeout=300):
    if task_type != "chief":
        return

    workers = tf_config["cluster"].get("worker", [])
    start_time = time.time()

    while True:
        all_ready = True

        for i, worker in enumerate(workers):
            host, _ = worker.split(":")
            readiness_port = BASE_READINESS_PORT + i + 1

            try:
                with socket.create_connection((host, readiness_port), timeout=5):
                    print(f"[READY] {host}:{readiness_port}", flush=True)

            except Exception:
                print(f"[WAITING] {host}:{readiness_port}", flush=True)
                all_ready = False

        if all_ready:
            print("All workers are ready.", flush=True)
            return

        if time.time() - start_time > timeout:
            raise TimeoutError("Workers did not become ready within 5 minutes.")

        time.sleep(10)


readiness_port = get_readiness_port(task_type, task_index)
start_readiness_server(readiness_port)
time.sleep(5)
wait_for_workers(tf_config)


# 4- Create TensorFlow Distributed Strategy for all workers
strategy = tf.distribute.MultiWorkerMirroredStrategy()

# 5- Dataset 
def make_dataset(shuffle=True):
    dataset = tf.data.experimental.make_csv_dataset(
        DATA_PATH,
        batch_size=BATCH_SIZE,
        label_name="label",
        num_epochs=None,
        shuffle=shuffle
    )

    def prepare(features, label):
        text = features["text"]
        label = tf.cast(label, tf.float32)
        return text, label

    dataset = dataset.map(prepare)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )

    return dataset.with_options(options)


train_dataset = make_dataset(shuffle=True)

adapt_dataset = tf.data.experimental.make_csv_dataset(
    DATA_PATH,
    batch_size=BATCH_SIZE,
    label_name="label",
    num_epochs=1,
    shuffle=False
).map(lambda features, label: features["text"])

# 6- Build Distributed:
#    A. Vectorize the text
#    B. Build the model
#    C. Compile the model      

with strategy.scope():
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE,
        output_mode="int",
        output_sequence_length=MAX_LEN
    )

    vectorizer.adapt(adapt_dataset)

    model = build_model(
        vectorizer=vectorizer,
        vocab_size=VOCAB_SIZE
    )

    optimizer = tf.keras.optimizers.Adam()

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

# 7- Fit the model
model.fit(
    train_dataset,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH
)


# 8- Save Model
os.makedirs(OUTPUT_PATH_MODEL, exist_ok=True)

if task_type == "chief":
    save_path = os.path.join(OUTPUT_PATH_MODEL, "model.keras")
else:
    save_path = os.path.join(
        OUTPUT_PATH_MODEL,
        f"worker_temp_{task_type}_{task_index}.keras"
    )

model.save(save_path)

print("Model saved to:", save_path, flush=True)

time.sleep(5)