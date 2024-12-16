import os
from dotenv import load_dotenv
load_dotenv('.env')

def is_public_domain(domain):
  """
  Checks if a string is a public internet domain name or a local network name.

  Args:
      domain: The string to be checked.

  Returns:
      True if the domain is a public internet domain name, False otherwise.
  """
  # Public domain names must have at least two dots (.) separating subdomains.
  if domain.count('.') < 1:
    return False

  # Local network names typically use private IP ranges (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16).
  try:
    # Attempt to convert the domain to an IP address.
    ip_address = int(domain.split('.')[0])
    if ip_address in range(10, 20) or ip_address in range(172, 174) or ip_address == 192:
      return False
  except ValueError:
    pass

  # Consider other TLDs (Top-Level Domains) besides the most common (.com, .net, .org).
  public_suffixes = [".com", ".net", ".org", ".info", ".biz", ".edu", ".gov", ".mil"]
  return any(domain.endswith(suffix) for suffix in public_suffixes)

# require apt-get install openssl and certbot
def generate_ssl_certificates():
    """generate ssl certficates for either localhost or specified domain name SSL_DOMAIN_NAME"""
    print('GEN CERTS')
    domain = os.environ.get('SSL_DOMAIN_NAME', 'localhost')
    email = os.environ.get('SSL_EMAIL', 'none@syntithenai.com')
    print(domain)
    cert_path = '/etc/letsencrypt/live/' + domain
    http_port = os.environ.get('SSL_HTTP_PORT', '19999')
    if not is_public_domain(domain):
        print('GEN LOCALHOST SSL KEY')
        os.system(' '.join(['mkdir', '-p', cert_path]))
        cmd = [
            'openssl',
            'req',
            '-x509',
            '-newkey',
            'rsa:4096',
            '-keyout',
            cert_path +
            '/privkey.pem',
            '-out',
            cert_path +
            '/cert.pem',
            '-days',
            '365',
            '-nodes',
            '-subj',
            '/CN=localhost']
        os.system(' '.join(cmd))

    else:
        # files exist so renew
        if os.path.isfile(cert_path + '/cert.pem') and os.path.isfile(cert_path + \
        '/fullchain.pem') and os.path.isfile(cert_path + '/privkey.pem'):
            print('RENEW CERTS')
            cmd = ['certbot', '	renew']
            print(cmd)
            os.system(' '.join(cmd))

        else:
            print('GENERATE CERTS')
            cmd = [
                'certbot',
                'certonly',
                '-a',
                'standalone',
                '--http-01-port', http_port,
                '--agree-tos',
                '-d',
                domain,
                '-m',
                email,
                ' --noninteractive']
            print(cmd)
            os.system(' '.join(cmd))

if __name__ == "__main__":
	generate_ssl_certificates()
