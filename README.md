from phrascriber import Client

receive_func = lambda x: print(x)
client = Client("10.0.0.2", "6969", receive_func)
client.run()

