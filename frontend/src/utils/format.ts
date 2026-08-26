export const formatPercent = (val: number) => `${(val * 100).toFixed(1)}%`;
export const formatMs = (val: number) => `${val.toFixed(0)}ms`;
export const formatNumber = (val: number) => new Intl.NumberFormat('en-US').format(val);
