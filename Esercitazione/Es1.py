import tensorflow as tf

physical_devices = tf.config.list_physical_devices()
print("Devices:", physical_devices)

details = tf.config.experimental.get_device_details(physical_devices[0])

print ("Details: ", details.get("compute_capability"))