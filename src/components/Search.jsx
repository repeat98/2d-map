// src/components/Search.jsx

import React, { useState, useEffect } from 'react';
import './Search.scss';
import PropTypes from 'prop-types';

const Search = ({ onSearch }) => {
  const [inputValue, setInputValue] = useState('');

  // Debounce the search input to optimize performance
  useEffect(() => {
    const handler = setTimeout(() => {
      onSearch(inputValue.trim());
    }, 100); // 300ms delay

    return () => {
      clearTimeout(handler);
    };
  }, [inputValue, onSearch]);

  return (
    <div className="search">
      <input
        type="text"
        id="searchInput"
        placeholder="Search tracks..."
        value={inputValue}
        onChange={(e) => setInputValue(e.target.value)}
        aria-label="Search tracks"
      />
    </div>
  );
};

Search.propTypes = {
  onSearch: PropTypes.func.isRequired,
};

export default Search;