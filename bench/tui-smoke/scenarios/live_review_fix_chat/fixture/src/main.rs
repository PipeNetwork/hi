//! Minimal IRC-style TCP chat. Integration tests spawn this binary.
//!
//! Intentionally buggy: `reply` never reaches the socket, and KICK does not
//! drop the target from the channel fan-out. Live e2e asks hi to find and fix
//! those defects against `cargo test --offline`.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

struct Subscriber {
    nick: String,
    tx: Sender<String>,
}

struct State {
    channels: HashMap<String, Vec<Subscriber>>,
    ops: HashMap<String, String>,
}

type Shared = Arc<Mutex<State>>;

fn main() {
    let addr = std::env::var("CHAT_ADDR").unwrap_or_else(|_| "127.0.0.1:0".to_string());
    let listener = TcpListener::bind(&addr).expect("bind chat listener");
    let bound = listener.local_addr().expect("local addr");
    println!("chat server listening on {bound}");
    let _ = std::io::stdout().flush();

    let state = Arc::new(Mutex::new(State {
        channels: HashMap::new(),
        ops: HashMap::new(),
    }));

    for incoming in listener.incoming() {
        let Ok(stream) = incoming else { continue };
        let state = Arc::clone(&state);
        thread::spawn(move || handle_connection(stream, state));
    }
}

fn handle_connection(stream: TcpStream, state: Shared) {
    let mut writer = stream.try_clone().expect("clone stream");
    let reader = BufReader::new(stream);
    let (tx, rx) = mpsc::channel::<String>();
    thread::spawn(move || {
        for msg in rx {
            if writer.write_all(msg.as_bytes()).is_err() || writer.flush().is_err() {
                break;
            }
        }
    });

    // BUG: replies are appended here and never copied onto `tx`, so Welcome
    // and OK/ERROR lines never reach the client. Broadcasts that use `tx`
    // still work, which is how KICKED can arrive after REGISTER/JOIN are
    // fixed without also fixing kick membership.
    let mut pending: Vec<String> = Vec::new();
    let reply = |pending: &mut Vec<String>, line: String| {
        pending.push(line);
    };

    reply(&mut pending, "Welcome to chat. Type HELP for commands.\n".into());

    let mut nick: Option<String> = None;
    let mut joined: Vec<String> = Vec::new();

    for line in reader.lines() {
        let Ok(line) = line else { break };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, ' ');
        let verb = parts.next().unwrap_or("").to_uppercase();
        let rest = parts.next().unwrap_or("").trim();
        match verb.as_str() {
            "REGISTER" | "LOGIN" => {
                let name = rest.split_whitespace().next().unwrap_or("");
                if name.is_empty() || nick.is_some() {
                    reply(&mut pending, "ERROR invalid\n".into());
                    continue;
                }
                nick = Some(name.to_string());
                reply(&mut pending, format!("OK registered as {name}\n"));
            }
            "JOIN" => {
                let Some(name) = nick.clone() else {
                    reply(&mut pending, "ERROR not logged in\n".into());
                    continue;
                };
                if rest.is_empty() {
                    reply(&mut pending, "ERROR missing channel\n".into());
                    continue;
                }
                let channel = rest.to_string();
                {
                    let mut st = state.lock().expect("state");
                    let subs = st.channels.entry(channel.clone()).or_default();
                    if !subs.iter().any(|s| s.nick == name) {
                        subs.push(Subscriber {
                            nick: name.clone(),
                            tx: tx.clone(),
                        });
                    }
                    st.ops.entry(channel.clone()).or_insert_with(|| name.clone());
                }
                if !joined.contains(&channel) {
                    joined.push(channel.clone());
                }
                reply(&mut pending, format!("OK joined {channel}\n"));
            }
            "PRIVMSG" => {
                let Some(name) = nick.clone() else {
                    reply(&mut pending, "ERROR not logged in\n".into());
                    continue;
                };
                let (channel, message) = match rest.split_once(' ') {
                    Some((ch, msg)) => (
                        ch.trim().to_string(),
                        msg.trim_start_matches(':').to_string(),
                    ),
                    None => {
                        reply(&mut pending, "ERROR missing message\n".into());
                        continue;
                    }
                };
                broadcast(&state, &channel, format!("PRIVMSG {channel} {name} :{message}\n"));
            }
            "KICK" => {
                let Some(name) = nick.clone() else {
                    reply(&mut pending, "ERROR not logged in\n".into());
                    continue;
                };
                let mut it = rest.split_whitespace();
                let Some(channel) = it.next() else {
                    reply(&mut pending, "ERROR missing channel\n".into());
                    continue;
                };
                let Some(target) = it.next() else {
                    reply(&mut pending, "ERROR missing user\n".into());
                    continue;
                };
                let allowed = {
                    let st = state.lock().expect("state");
                    st.ops.get(channel).map(|op| op == &name).unwrap_or(false)
                };
                if !allowed {
                    reply(&mut pending, "ERROR not an operator\n".into());
                    continue;
                }
                reply(&mut pending, format!("OK kicked {target}\n"));
                broadcast(&state, channel, format!("KICK {channel} {target}\n"));
                broadcast(&state, channel, format!("KICKED {channel}\n"));
                // BUG: the target stays subscribed, so later PRIVMSG still
                // arrives on their socket.
            }
            "QUIT" => break,
            "PING" => reply(&mut pending, "PONG\n".into()),
            _ => reply(&mut pending, "ERROR unknown command\n".into()),
        }
    }

    if let Some(name) = nick {
        let mut st = state.lock().expect("state");
        for channel in &joined {
            if let Some(subs) = st.channels.get_mut(channel) {
                subs.retain(|s| s.nick != name);
            }
        }
    }
}

fn broadcast(state: &Shared, channel: &str, line: String) {
    let mut st = state.lock().expect("state");
    if let Some(subs) = st.channels.get_mut(channel) {
        subs.retain(|s| s.tx.send(line.clone()).is_ok());
    }
}
