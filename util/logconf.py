import logging

# Before we define our own handler, we get rid of any possible root logger handlers already present
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
for handler in list(root_logger.handlers):
    root_logger.removeHandler(handler)

# set format
logfmt_str = "%(asctime)s %(levelname)-8s pid: %(process)d %(name)s : %(lineno)03d : %(funcName)s %(message)s"
formatter = logging.Formatter(logfmt_str)

# create handler
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)
streamHandler.setLevel(logging.DEBUG)

# add handler to root, so it's available for any file importing this file
root_logger.addHandler(streamHandler)
