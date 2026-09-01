//! Fail-closed URL checks for `browser_exec` navigation.

use std::net::{IpAddr, SocketAddr, ToSocketAddrs};

use anyhow::{Result, bail};
use url::Url;

#[derive(Clone, Copy, Debug, Default)]
pub struct BrowserPolicy {
    pub allow_private: bool,
}

/// Hosts that are never navigable, even with `allow_private`.
fn is_blocked_metadata_host(host: &str) -> bool {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    host == "metadata.google.internal"
        || host == "metadata.google.internal."
        || host.ends_with(".metadata.google.internal")
}

fn ipv6_as_v4(v6: std::net::Ipv6Addr) -> Option<std::net::Ipv4Addr> {
    if let Some(v4) = v6.to_ipv4_mapped().or_else(|| v6.to_ipv4()) {
        return Some(v4);
    }
    let segments = v6.segments();
    if segments[0] == 0x2002 {
        return Some(std::net::Ipv4Addr::new(
            (segments[1] >> 8) as u8,
            segments[1] as u8,
            (segments[2] >> 8) as u8,
            segments[2] as u8,
        ));
    }
    if segments[..6] == [0x0064, 0xff9b, 0, 0, 0, 0] {
        return Some(std::net::Ipv4Addr::new(
            (segments[6] >> 8) as u8,
            segments[6] as u8,
            (segments[7] >> 8) as u8,
            segments[7] as u8,
        ));
    }
    None
}

fn is_always_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
                || matches!(v4.octets(), [169, 254, 169, 254] | [100, 100, 100, 200])
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = ipv6_as_v4(v6) {
                return is_always_blocked_ip(IpAddr::V4(v4));
            }
            let segments = v6.segments();
            v6.is_unspecified()
                || v6.is_unicast_link_local()
                || v6.is_multicast()
                // EC2 Instance Metadata Service IPv6 endpoint. This remains
                // blocked even when ordinary private-network access is opted in.
                || segments == [0xfd00, 0x0ec2, 0, 0, 0, 0, 0, 0x0254]
                // Teredo embeds an obfuscated IPv4 endpoint and makes safe
                // classification needlessly ambiguous; it is never required
                // for ordinary browser navigation.
                || (segments[0] == 0x2001 && segments[1] == 0)
                // RFC 8215 local-use NAT64 prefix.
                || segments[..3] == [0x0064, 0xff9b, 0x0001]
        }
    }
}

fn is_private_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            let [a, b, c, _] = v4.octets();
            v4.is_loopback()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_private()
                || v4.is_broadcast()
                || v4.is_documentation()
                || a == 0
                || (a == 100 && (64..=127).contains(&b))
                || (a == 192 && b == 0 && c == 0)
                || (a == 192 && b == 88 && c == 99)
                || (a == 198 && (18..=19).contains(&b))
                || a >= 224
        }
        IpAddr::V6(v6) => {
            if let Some(v4) = ipv6_as_v4(v6) {
                return is_private_ip(IpAddr::V4(v4));
            }
            let segments = v6.segments();
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_unicast_link_local()
                || v6.is_multicast()
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfec0
                || segments[..4] == [0x0100, 0, 0, 0]
                || (segments[0] == 0x2001 && segments[1] == 0x0002)
                || (segments[0] == 0x2001 && segments[1] == 0x0db8)
        }
    }
}

/// Synchronous URL check used by tests and before CDP navigation.
pub fn check_navigation_url(url: &str, policy: BrowserPolicy) -> Result<()> {
    let parsed = Url::parse(url).map_err(|e| anyhow::anyhow!("invalid URL '{url}': {e}"))?;
    let scheme = parsed.scheme();
    if scheme != "http" && scheme != "https" {
        bail!("browser_exec requires http:// or https:// URLs (got '{scheme}')");
    }
    match parsed.host() {
        Some(url::Host::Ipv4(v4)) => check_ip(IpAddr::V4(v4), policy),
        Some(url::Host::Ipv6(v6)) => check_ip(IpAddr::V6(v6), policy),
        Some(url::Host::Domain(host)) => {
            if is_blocked_metadata_host(host) {
                bail!("refused: '{host}' is a cloud metadata hostname");
            }
            Ok(())
        }
        None => bail!("URL '{url}' has no host"),
    }
}

/// Parse, block metadata/private literals, then DNS-resolve the host (fail closed).
/// Call this before launching Chrome so a blocked host never starts a browser.
pub fn check_url_with_dns(url: &str, policy: BrowserPolicy) -> Result<()> {
    check_navigation_url(url, policy)?;
    let parsed = Url::parse(url).map_err(|e| anyhow::anyhow!("invalid URL '{url}': {e}"))?;
    if let Some(host) = parsed.host_str() {
        resolve_and_check_host(host, policy)?;
    }
    Ok(())
}

/// Refuse resolved addresses that are metadata, link-local, or (unless opted in) private.
pub fn check_resolved_ips(host: &str, ips: &[IpAddr], policy: BrowserPolicy) -> Result<()> {
    if ips.is_empty() {
        bail!("could not resolve host '{host}'");
    }
    for ip in ips {
        if is_always_blocked_ip(*ip) {
            bail!("refused: '{host}' resolves to blocked metadata/link-local {ip}");
        }
        if is_private_ip(*ip) && !policy.allow_private {
            bail!(
                "refused: '{host}' resolves to private/loopback {ip} (set [browser] allow_private_urls = true)"
            );
        }
    }
    Ok(())
}

fn check_ip(ip: IpAddr, policy: BrowserPolicy) -> Result<()> {
    if is_always_blocked_ip(ip) {
        bail!("refused: '{ip}' is a blocked metadata/link-local address");
    }
    if is_private_ip(ip) && !policy.allow_private {
        bail!(
            "refused: '{ip}' is a private/loopback address (set [browser] allow_private_urls = true)"
        );
    }
    Ok(())
}

/// Resolve `host` and refuse private/metadata addresses. DNS errors fail closed.
pub fn resolve_and_check_host(host: &str, policy: BrowserPolicy) -> Result<()> {
    resolve_and_check_host_ips(host, policy).map(|_| ())
}

/// Resolve once and return only addresses that passed the policy. Callers that
/// perform the network request must pin their transport to this exact set;
/// resolving the hostname again would reopen a DNS-rebinding window.
pub(crate) fn resolve_and_check_host_ips(host: &str, policy: BrowserPolicy) -> Result<Vec<IpAddr>> {
    if is_blocked_metadata_host(host) {
        bail!("refused: '{host}' is a cloud metadata hostname");
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        if is_always_blocked_ip(ip) {
            bail!("refused: '{host}' is a blocked metadata/link-local address");
        }
        if is_private_ip(ip) && !policy.allow_private {
            bail!("refused: '{host}' is a private/loopback address");
        }
        return Ok(vec![ip]);
    }
    let addrs: Vec<SocketAddr> = match (host, 0u16).to_socket_addrs() {
        Ok(iter) => iter.collect(),
        Err(err) => bail!("could not resolve host '{host}': {err}"),
    };
    let ips: Vec<IpAddr> = addrs.iter().map(|socket| socket.ip()).collect();
    check_resolved_ips(host, &ips, policy)?;
    Ok(ips)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_metadata_and_link_local() {
        let closed = BrowserPolicy {
            allow_private: false,
        };
        assert!(check_navigation_url("http://169.254.169.254/latest", closed).is_err());
        assert!(check_navigation_url("http://metadata.google.internal/", closed).is_err());
        assert!(check_navigation_url("http://[::ffff:169.254.169.254]/", closed).is_err());
        assert!(check_navigation_url("ftp://example.com/", closed).is_err());
    }

    #[test]
    fn private_requires_opt_in() {
        let closed = BrowserPolicy {
            allow_private: false,
        };
        let open = BrowserPolicy {
            allow_private: true,
        };
        assert!(check_navigation_url("http://127.0.0.1:3000/", closed).is_err());
        assert!(check_navigation_url("http://127.0.0.1:3000/", open).is_ok());
        assert!(check_navigation_url("http://192.168.1.10/", open).is_ok());
        // Metadata stays blocked even when private URLs are allowed.
        assert!(check_navigation_url("http://169.254.169.254/", open).is_err());
        assert!(check_navigation_url("http://[fd00:ec2::254]/", open).is_err());
    }

    #[test]
    fn public_https_is_ok() {
        check_navigation_url("https://example.com/login", BrowserPolicy::default()).unwrap();
    }

    #[test]
    fn resolve_blocks_metadata_hostname_without_dns() {
        let err = resolve_and_check_host("metadata.google.internal", BrowserPolicy::default())
            .expect_err("metadata host");
        assert!(err.to_string().contains("cloud metadata"), "{err}");
    }

    #[test]
    fn dns_to_link_local_metadata_is_blocked() {
        let ip: IpAddr = "169.254.169.254".parse().unwrap();
        let err = check_resolved_ips("evil.example", &[ip], BrowserPolicy::default())
            .expect_err("link-local metadata");
        assert!(err.to_string().contains("169.254.169.254"), "{err}");
        // Even with private URLs allowed.
        let err = check_resolved_ips(
            "evil.example",
            &[ip],
            BrowserPolicy {
                allow_private: true,
            },
        )
        .expect_err("metadata stays blocked");
        assert!(err.to_string().contains("blocked metadata"), "{err}");
    }

    #[test]
    fn restricted_policy_rejects_all_non_global_addresses() {
        let carrier_grade_nat: IpAddr = "100.64.0.1".parse().unwrap();
        assert!(
            check_resolved_ips(
                "rebind.example",
                &[carrier_grade_nat],
                BrowserPolicy::default()
            )
            .is_err()
        );
    }

    #[test]
    fn private_override_still_blocks_non_unicast_and_known_metadata() {
        let open = BrowserPolicy {
            allow_private: true,
        };
        for ip in ["224.0.0.1", "ff02::1", "100.100.100.200", "fd00:ec2::254"] {
            let ip: IpAddr = ip.parse().unwrap();
            assert!(check_resolved_ips("blocked.example", &[ip], open).is_err());
        }
    }

    #[test]
    fn transition_addresses_cannot_hide_private_or_metadata_ipv4() {
        let closed = BrowserPolicy::default();
        let open = BrowserPolicy {
            allow_private: true,
        };
        for address in [
            "2002:7f00:0001::1",
            "64:ff9b::7f00:1",
            "2002:a9fe:a9fe::1",
            "64:ff9b::a9fe:a9fe",
        ] {
            let ip: IpAddr = address.parse().unwrap();
            assert!(check_resolved_ips("transition.example", &[ip], closed).is_err());
        }
        for address in ["2002:a9fe:a9fe::1", "64:ff9b::a9fe:a9fe"] {
            let ip: IpAddr = address.parse().unwrap();
            assert!(
                check_resolved_ips("transition.example", &[ip], open).is_err(),
                "embedded link-local metadata must remain blocked with the override"
            );
        }
    }

    #[test]
    fn additional_non_global_ranges_are_rejected() {
        for address in [
            "0.1.2.3",
            "192.88.99.1",
            "198.18.0.1",
            "fec0::1",
            "100::1",
            "2001:2::1",
            "2001:db8::1",
        ] {
            let ip: IpAddr = address.parse().unwrap();
            assert!(
                check_resolved_ips("special.example", &[ip], BrowserPolicy::default()).is_err(),
                "{address} must not be treated as public"
            );
        }
    }
}
