"""Typed staff-only digest: counts, anonymous customer IDs, ranks and a test link.

No credentials, customer labels, scores or explanations are accepted. This module
only constructs and validates previews, it never connects to LINE.
"""
from dataclasses import dataclass
import ipaddress
import math
from numbers import Real
import re
from urllib.parse import urlsplit, urlunsplit

DEFAULT_PLATFORM_URL = 'http://localhost:8501/'
PREVIEW_LIMIT = 10  # Message preview only, never a change to the policy's list.


@dataclass(frozen=True, slots=True)
class DigestEntry:
    customer_id: int
    risk_rank: int


@dataclass(frozen=True, slots=True)
class StaffDigest:
    list_count: int
    entries: tuple[DigestEntry, ...]
    platform_url: str = DEFAULT_PLATFORM_URL


def validate_platform_url(value):
    """Allow loopback, private test LANs and reserved .test names, never a public URL.

    Only the root page is linked. Credentials, queries, fragments, encoded paths
    and redirect/deep-link parameters are deliberately not accepted.
    """
    if not isinstance(value, str) or not value or any(c.isspace() for c in value) or '\\' in value:
        raise ValueError('平台連結只允許本機或測試環境')
    try:
        parsed = urlsplit(value)
        host = (parsed.hostname or '').lower().rstrip('.')
        port = parsed.port
        if (parsed.scheme not in ('http', 'https') or not host or len(host) > 253 or parsed.username is not None
                or parsed.password is not None or parsed.query or parsed.fragment
                or parsed.path not in ('', '/') or port is not None and not 1 <= port <= 65535):
            raise ValueError
        allowed = host == 'localhost'
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            labels = host.split('.')
            allowed = allowed or (host.endswith('.test') and all(
                re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?', label) for label in labels))
        else:
            private_ranges = ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16', 'fc00::/7')
            allowed = address.is_loopback or any(address in ipaddress.ip_network(network) for network in private_ranges)
        if not allowed:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError('平台連結只允許本機或測試環境，不能使用公開網址或附加參數') from None
    return urlunsplit((parsed.scheme, parsed.netloc, '/', '', ''))


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def render_digest(digest):
    """Revalidate even an already-built object, return the exact preview/send text."""
    if type(digest) is not StaffDigest or not _integer(digest.list_count, 0, 30000):
        raise ValueError('行員摘要必須使用固定欄位的 StaffDigest')
    if (not isinstance(digest.entries, tuple) or len(digest.entries) > PREVIEW_LIMIT
            or len(digest.entries) > digest.list_count or (digest.list_count > 0 and not digest.entries)):
        raise ValueError('行員摘要的名單人數或摘要範圍不符')
    ids, ranks = [], []
    for entry in digest.entries:
        if (type(entry) is not DigestEntry or not _integer(entry.customer_id, 1, 30000)
                or not _integer(entry.risk_rank, 1, 30000)):
            raise ValueError('行員摘要只允許數值客戶 ID 與排名')
        ids.append(entry.customer_id); ranks.append(entry.risk_rank)
    if len(set(ids)) != len(ids) or len(set(ranks)) != len(ranks) or ranks != sorted(ranks):
        raise ValueError('行員摘要的客戶 ID 或排名重複，或未依排名排列')
    url = validate_platform_url(digest.platform_url)
    lines = [f'預警名單人數：{digest.list_count}']
    lines += [f'客戶 ID：{entry.customer_id}，排名：{entry.risk_rank}' for entry in digest.entries]
    lines.append(f'平台連結：{url}')
    return '\n'.join(lines)


def build_digest(frame, *, platform_url=DEFAULT_PLATFORM_URL):
    """Use handoff in_list/risk_rank, preview first ten, retain the full list count.

    Input is the existing page DataFrame. Only these three columns are read,
    neither labels nor risk factors are copied into the immutable StaffDigest.
    """
    if not {'customer_id', 'risk_rank', 'in_list'} <= set(frame):
        raise ValueError('交接資料缺少名單摘要必要欄位')
    if not frame['in_list'].isin([0, 1]).all():
        raise ValueError('交接資料的名單標記不完整')
    selected = frame.loc[frame['in_list'].eq(1), ['customer_id', 'risk_rank']].copy()
    for column in ('customer_id', 'risk_rank'):
        values = selected[column]
        if values.isna().any() or not values.map(lambda value: isinstance(value, Real) and not isinstance(value, bool)
                and math.isfinite(value) and value == int(value) and 1 <= value <= 30000).all():
            raise ValueError('交接資料的客戶 ID 與排名必須為完整整數')
        if values.duplicated().any():
            raise ValueError('交接資料的客戶 ID 或排名重複')
    selected = selected.sort_values('risk_rank').head(PREVIEW_LIMIT)
    digest = StaffDigest(int(frame['in_list'].eq(1).sum()), tuple(
        DigestEntry(int(row.customer_id), int(row.risk_rank)) for row in selected.itertuples(index=False)),
        platform_url)
    render_digest(digest)
    return digest


if __name__ == '__main__':
    print('CLI 不提供發送功能。請在平台預覽行員名單摘要。')
