import logging
import socket


class Actor:
    def __init__(self, host, port, packets_size=1024, sentinel=b'BREAK'):
        """

        :type host: str
        :type port: int
        :type packets_size: int, better be a power of 2
        :type sentinel: binary
        :param host: ipv4 address of the host. If nobody is listening on this address, this actor will create a server on local_host
        :param port: port number. If nobody is listening on this address, this actor will create a server on this port
        :param packets_size: number of bytes that will be sent and received per packet. The Actor will then ask for the next packet of the data, until it receives the sentinel.
        :param sentinel: signal transmitted to inform that the data has been fully transmitted.
        """
        self.logger = logging.getLogger(__name__)
        self.sentinel = sentinel
        self.packets_size = packets_size
        self.address = host + ' : ' + str(port)
        self.host = host
        self.port = port

        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.logger.info(str(self.socket.recv(1024)))
        except socket.error:
            self.logger.info("Nobody is listening, binding a server...")
            conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            conn.bind(('', self.port))
            conn.listen(1)
            self.socket, socket_address = conn.accept()
            self.logger.info('Connected by '+ socket_address[0]+':'+str(socket_address[1]))
            self.socket.send(b'Connection accepted')

    def __packet_reception(self):
        self.socket.send(b'next')
        return self.socket.recv(self.packets_size)

    def reception(self):
        msg=b''.join(iter(self.__packet_reception, self.sentinel))
        if msg == b'KILL': self.close()
        return msg

    def transmission(self, data):
        """

        :type data: binary to be send
        """
        nb_bytes_send = 0
        for n in range(len(data) // self.packets_size + 1):
            if self.socket.recv(self.packets_size) == b'next':
                nb_bytes_send += self.socket.send(data[n * self.packets_size: (n + 1) * self.packets_size])
        if self.socket.recv(self.packets_size) == b'next':
            self.socket.send(self.sentinel)

        return nb_bytes_send

    def close(self):
        self.socket.close()
